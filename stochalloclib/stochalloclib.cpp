/*
 * Author: Liran Funaro <liran.funaro@gmail.com>
 *
 * Copyright (C) 2006-2018 Liran Funaro
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <string>
#include <cstdint>
#include <vector>
#include <queue>
#include <memory>
#include <algorithm>
#include <limits>

using uint32 = uint32_t;
using float32 = float;

#include "mat.hpp"
#include "debug.h"

typedef mat<float32, 2> ndarray_2d;


typedef enum {
    RESERVED=0, SHARES=1
} Status;

typedef double VirtualTime;

typedef struct {
    Status status;
    VirtualTime vtime;
    unsigned int id;
} Item;
    
class ItemGreater {
public:
    bool operator()(const Item* a, const Item* b) const {
        if (a->status == b->status)
            return a->vtime > b->vtime;
        else
            return a->status > b->status;
    }
};


typedef std::vector<Item> ItemVec;
typedef std::priority_queue<Item*, std::vector<Item*>, ItemGreater> ItemQueue;
typedef std::vector<ndarray_2d> AllocStepFunc;

class CFSQ {
    ItemQueue q;
    VirtualTime minVtime;

public:
    CFSQ() : minVtime(0) {}

    void push(Item& e) {
        e.vtime += minVtime;
        q.push(&e);
        minVtime = q.top()->vtime;
    }

    Item& pop() {
        Item* e = q.top();
        q.pop();
        if (!empty())
            minVtime = q.top()->vtime;

        e->vtime -= minVtime;
        return *e;
    }

    bool empty() const {
        return q.empty();
    }

    VirtualTime minV() const {
        return minVtime;
    }
};
        

float negligible_epoch_ratio = 1e-6;
float shares_eps = 1e-8;


class CFS {
    uint32 ndim;
    uint32 n;

    const float32* total_resources;
    const float32* resources_epoch;
    std::vector<float32> resource_eps;

    ndarray_2d required_resources;
    std::vector<ndarray_2d> shares;
    ndarray_2d limit;
    ndarray_2d alloc;

    const AllocStepFunc& alloc_step_funcs;

    std::vector<ItemVec> p;
    std::vector<CFSQ> q;

    std::vector<float32> unused_resources;
    std::vector<unsigned int> cur_step;
    std::vector<unsigned int> step_counter;

public:
    CFS(uint32 ndim, uint32 n,
        const float32* total_resources,
        const float32* resources_epoch,
        const ndarray_2d& required_resources,
        const ndarray_2d& reserved,
        const ndarray_2d& shares,
        const ndarray_2d& limit,
        const AllocStepFunc& alloc_step_funcs,
        const ndarray_2d& ret_alloc)
    : ndim(ndim), n(n),
    total_resources(total_resources),
    resources_epoch(resources_epoch),
    resource_eps(ndim),
    required_resources(required_resources),
    shares(),
    limit(limit),
    alloc(ret_alloc),
    alloc_step_funcs(alloc_step_funcs),
    p(ndim), q(ndim),
    unused_resources(total_resources, total_resources + ndim),
    cur_step(n, 0),
    step_counter(n, 0)
    {
        for (unsigned d=0; d<ndim; d++) {
            p[d].resize(n);
            for (unsigned i=0; i<n; i++) {
                p[d][i].status = RESERVED;
                p[d][i].vtime = 0;
                p[d][i].id = i;
                q[d].push(p[d][i]);
            }

            resource_eps[d] = resources_epoch[d] * negligible_epoch_ratio;
        }

        this->shares.push_back(reserved);
        this->shares.push_back(shares);
    }


    bool isResourceAllocatable(unsigned d) const {
        return !q[d].empty() && unused_resources[d] > resource_eps[d];
    }

    int chooseResource() const {
        int maxD = -1;
        float32 mostUnused = 0;
        for (unsigned d=0; d<ndim; d++) {
            if (!isResourceAllocatable(d))
                continue;

            float unused_resources_ratio = unused_resources[d] / total_resources[d];
            if (unused_resources_ratio > mostUnused) {
                maxD = d;
                mostUnused = unused_resources_ratio;
            }
        }
        return maxD;
    }

    void allocate() {
        while (true) {
            int d = chooseResource();
            if (d < 0)
                break;

            alloc_resource((unsigned)d);
        }
    }

    void alloc_resource(unsigned d) {
        Item& e = q[d].pop();

        auto epoch = resources_epoch[d];
        auto eps = resource_eps[d];

        unsigned int ind[] = {e.id, d};
        auto p_share = shares[e.status][ind];
        auto p_alloc = alloc[ind];
        auto p_req = required_resources[ind];
        auto p_limit = std::min(p_req, limit[ind]);

        unsigned int step_ind[] = {cur_step[e.id], d};
        auto& p_step_func = alloc_step_funcs[e.id];
        auto p_step = p_step_func[step_ind];

        bool step_barrier = false;

        // Stop handling this item without returning to the queue if:
        //  - Reached the required allocation.
        //  - Passed the reserved quantity and does not have shares.
        if (p_alloc + eps > p_limit || (e.status == SHARES && p_share < shares_eps))
            return;

        // Limit the epoch to the maximal required quantity.
        epoch = std::min(epoch, p_limit - p_alloc + eps);

        // If this resource finish this step and the epoch is larger than
        // the reminder, then move to the next step or reach a step barrier.
        // This loop will have maximum 2 iteration unless ndim==1.
        while (p_alloc + epoch + eps > p_step) {
            step_counter[e.id] = (step_counter[e.id] + 1) % ndim;
            if (step_counter[e.id] == 0 && step_ind[0] < p_step_func.size[0]-1) {
                step_ind[0]++;
                cur_step[e.id] = step_ind[0];
                p_step = p_step_func[step_ind];

                // Add player to the other resources queue.
                for(unsigned dd=0; dd<ndim; dd++) {
                    if (dd == d)
                        continue;
                    q[dd].push(p[dd][e.id]);
                }
            } else {
                epoch = p_step - p_alloc + eps;
                step_barrier = true;
                break;
            }
        }

        DEBUG_OUTPUT_CLEAN(e.id << "," << d << "," << unused_resources[d]
                   << "," << epoch
                   << "," << p_alloc
                   << "," << p_req
                   << "," << p_limit
                   << "," << e.status
                   << "," << p_share
                   << "," << std::boolalpha << step_barrier
                   << "," << step_counter[e.id]
                   << "," << step_ind[0]
                   << "," << p_step);

        // Update the allocation.
        p_alloc += epoch;
        alloc[ind] = p_alloc;
        unused_resources[d] -= epoch;

        // Increment virtual time.
        // Might need to switch from reserved to shares.
        if (e.status == RESERVED && p_alloc + eps > p_share) {
            auto shares_epoch = p_alloc - p_share;

            e.vtime = 0;

            e.status = SHARES;
            p_share = shares[e.status][ind];

            if(shares_epoch > eps && p_share > shares_eps)
                e.vtime += shares_epoch / p_share;
        } else {
            e.vtime += epoch / p_share;
        }

        DEBUG_OUTPUT_CLEAN(e.id << "," << d << "," << unused_resources[d]
           << "," << epoch
           << "," << p_alloc
           << "," << p_req
           << "," << p_limit
           << "," << e.status
           << "," << p_share
           << "," << std::boolalpha << step_barrier
           << "," << step_counter[e.id]
           << "," << step_ind[0]
           << "," << p_step);

        // Add the item back to the queue.
        if (p_alloc + eps < p_limit && !step_barrier && !(e.status == SHARES && p_share < shares_eps))
            q[d].push(e);
    }
};


// Shared library interface
extern "C" {

void cfs(uint32 ndim, uint32 n,
         float32* _total_resources,    // ndarray[np.float32, ndim=1]
         float32* _resources_epoch,    // ndarray[np.float32, ndim=1]
         float32* _required_resources, // ndarray[np.float32, ndim=2]
         float32* _reserved,           // ndarray[np.float32, ndim=2]
         float32* _shares,             // ndarray[np.float32, ndim=2]
         float32* _limit,              // ndarray[np.float32, ndim=2]
         float32* _alloc_step_funcs,   // ndarray[np.float32, ndim=2]
         uint32* _alloc_step_len,      // ndarray[np.uint32,  ndim=2]
         float32* _ret_alloc)          // ndarray[np.float32, ndim=2]
{
    uint32 sz_2d[] = {n, ndim};

    AllocStepFunc alloc_step_funcs;
    for(unsigned i=0; i<n; i++) {
        uint32 sz_step[] = {_alloc_step_len[i], ndim};
        auto step_func = ndarray_2d(_alloc_step_funcs, sz_step);
        alloc_step_funcs.push_back(step_func);
        _alloc_step_funcs += step_func.total_size();
    }

    CFS c(ndim, n, _total_resources, _resources_epoch,
            ndarray_2d(_required_resources, sz_2d),
            ndarray_2d(_reserved, sz_2d),
            ndarray_2d(_shares, sz_2d),
            ndarray_2d(_limit, sz_2d),
            alloc_step_funcs,
            ndarray_2d(_ret_alloc, sz_2d)
        );
    c.allocate();
}


void allocate_players_to_servers(uint32 seed, uint32 ndim, uint32 n, uint32 max_servers, uint32 niter,
                                 uint32 return_groups_count,
                                 float32* total_resources, float32* reserved, float32* shares,
                                 float32* ret_allocated_count, float32* ret_allocated_resources, uint32* ret_active,
                                 int32_t* ret_groups) {
    // total_resources         [ndim]
    // reserved, shares        [n, ndim]
    // ret_allocated_count     [max_servers]
    // ret_allocated_resources [max_servers, ndim * 2]
    // ret_active              [max_servers]
    // ret_groups              [return_groups_count, n]
    // ret_* assumed to initialized with zeros
    // ret_groups can be NULL
    float32 eps = std::numeric_limits<float32>::epsilon();
    float32 max_flt = std::numeric_limits<float32>::max();
    std::srand ( unsigned (seed) );

    std::vector<float32> cur_allocated(max_servers * ndim * 2);

    std::vector<unsigned int> order(n);
    for (unsigned p=0; p<n; p++)
        order[p] = p;

    // Initialize have reserved array
    std::vector<bool> have_reserved(n);
    for (unsigned int p=0; p<n; p++) {
        have_reserved[p] = false;
        float32* cur_p = &reserved[p*ndim];
        for (unsigned int r=0; r<ndim; r++) {
            if (cur_p[r] > eps) {
                have_reserved[p] = true;
                break;
            }
        }
    }

    for (unsigned int cur_iter=0; cur_iter<niter; cur_iter++) {
        std::random_shuffle(order.begin(), order.end());
        unsigned int max_active = 0;
        int32_t* cur_groups = NULL;
        if (cur_iter < return_groups_count)
            cur_groups = &ret_groups[cur_iter*n];

        // Initialize servers available resources and active
        for (unsigned int cur_server=0; cur_server < max_servers; cur_server++) {
            float32* cur_s = &cur_allocated[cur_server*ndim*2];
            for (unsigned r=0; r<ndim*2; r++)
                cur_s[r] = 0;
        }

        // Allocate players with reserved to the first fitting server
        for (unsigned int order_index=0; order_index<n; order_index++) {
            auto player_index = order[order_index];
            if (!have_reserved[player_index])
                continue;

            float32* cur_p = &reserved[player_index*ndim];
            float32* cur_p_shares = &shares[player_index*ndim];

            bool fit = false;
            for (unsigned int cur_server=0; cur_server < max_servers && !fit; cur_server++) {
                float32* cur_s = &cur_allocated[cur_server*ndim*2];
                fit = true;
                for (unsigned int r=0; r<ndim; r++) {
                    if (cur_s[r] + cur_p[r] > total_resources[r]) {
                        fit = false;
                        break;
                    }
                }

                if (fit) {
                    max_active = max_active < cur_server ? cur_server : max_active;
                    ret_allocated_count[cur_server] += 1;
                    if (cur_groups != NULL)
                        cur_groups[player_index] = cur_server;
                    for (unsigned int r=0; r<ndim; r++) {
                        cur_s[r] += cur_p[r];
                        cur_s[ndim + r] += cur_p_shares[r];
                    }
                }
            }

            if (!fit && cur_groups != NULL)
                cur_groups[player_index] = -1;
        }

        // Allocate players with only shares
        for (unsigned int order_index=0; order_index<n; order_index++) {
            auto player_index = order[order_index];
            if (have_reserved[player_index])
                continue;

            unsigned int min_server = 0;
            float32 min_value = max_flt;
            // We don't put share players in the last server, otherwise most of them will be allocated to it
            for (unsigned int cur_server=0; cur_server < max_active; cur_server++) {
                float32* cur_s = &cur_allocated[cur_server*ndim*2];
                float32 cur_value = 0.;
                for (unsigned int r=0; r<ndim; r++)
                    cur_value += cur_s[ndim + r] * cur_s[ndim + r];

                if (cur_value < min_value) {
                    min_value = cur_value;
                    min_server = cur_server;
                }
            }

            float32* cur_p_shares = &shares[player_index*ndim];
            float32* cur_s = &cur_allocated[min_server*ndim*2];
            ret_allocated_count[min_server] += 1;
            for (unsigned int r=0; r<ndim; r++)
                cur_s[ndim + r] += cur_p_shares[r];

            if (cur_groups != NULL)
                cur_groups[player_index] = min_server;
        }

        // Increment active servers
        for (unsigned int cur_server=0; cur_server <= max_active; cur_server++) {
            ret_active[cur_server] += 1;
        }

        // Add iteration data to ret_allocated
        for (unsigned int cur_server=0; cur_server <= max_active; cur_server++) {
            float32* cur_s = &cur_allocated[cur_server*ndim*2];
            float32* cur_ret_alloc = &ret_allocated_resources[cur_server*ndim*2];
            for (unsigned int r=0; r<ndim*2; r++)
                cur_ret_alloc[r] += cur_s[r];
        }
    }

    for (unsigned int cur_server=0; cur_server < max_servers; cur_server++) {
        if (ret_active[cur_server] == 0)
            continue;

        ret_allocated_count[cur_server] /= float32(ret_active[cur_server]);

        float32* cur_ret_alloc = &ret_allocated_resources[cur_server*ndim*2];
        for (unsigned int r=0; r<ndim*2; r++)
            cur_ret_alloc[r] /= float32(ret_active[cur_server]);
    }
}

} // extern "C"
