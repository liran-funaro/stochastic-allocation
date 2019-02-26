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

#ifndef VEC_HPP_
#define VEC_HPP_

#include <numeric>
#include <limits>
#include <cmath>

#define FOR_EACH_DIM_D(__var__, __dim__) \
    for (unsigned int __var__=0; (__var__) < (__dim__); (__var__)++)

#define FOR_EACH_DIM_D_REV(__var__, __dim__) \
    for (unsigned int __var__=(__dim__)-1; (__var__) != static_cast<unsigned int>(-1); (__var__)--)

#define FOR_EACH_DIM(__var__) FOR_EACH_DIM_D(__var__, D)
#define FOR_EACH_DIM_REV(__var__) FOR_EACH_DIM_D_REV(__var__, D)


template <typename T>
static inline T get_nextafter(T t) {
    if (std::is_integral<T>::value)
        return t + 1;
    else
        return std::nextafter(t, std::numeric_limits<T>::max());
}


template <unsigned int D, typename T = uint32_t>
class vec {
    static_assert(std::is_arithmetic<T>::value, "Type T must be numeric");

public:
    static const unsigned int dim = D;

private:
    T v[D];

public:
    vec() {}

    vec(T i) {
        FOR_EACH_DIM(d)
            v[d] = i;
    }

    template<typename VEC_TYPE>
    vec(const VEC_TYPE& other) {
        FOR_EACH_DIM(i)
            v[i] = other[i];
    }

    /***************************************************************************
     * Access and set operators
     **************************************************************************/

    inline T operator[](int index) const { return v[index]; }
    inline T& operator[](int index) { return v[index]; }

    inline vec<D,T>& operator=(const vec<D,T>& other) {
        FOR_EACH_DIM(i)
            v[i] = other[i];
        return *this;
    }

    inline vec<D,T>& operator=(const T* other) {
        FOR_EACH_DIM(i)
            v[i] = other[i];
        return *this;
    }

    inline void reset() {
        FOR_EACH_DIM(i)
            v[i] = 0;
    }

    inline unsigned int size() const {
        unsigned int ret = 1;
        FOR_EACH_DIM(i)
            ret *= v[i];
        return ret;
    }

    /***************************************************************************
     * Unary operators
     **************************************************************************/

    inline void dec_1() {
        FOR_EACH_DIM(i)
            v[i]--;
    }

    inline void inc_1() {
        FOR_EACH_DIM(i)
            v[i]++;
    }

    inline void setall(const T c) {
        FOR_EACH_DIM(i)
            v[i] = c;
    }

    inline void nextafter() {
        FOR_EACH_DIM(i)
            v[i] = get_nextafter(v[i]);
    }

    /***************************************************************************
     * Binary operators
     **************************************************************************/

    inline void min(const vec<D,T>& other) {
        FOR_EACH_DIM(i) {
            if (v[i] > other[i])
                v[i] = other[i];
        }
    }

    inline void max(const vec<D,T>& other) {
        FOR_EACH_DIM(i) {
            if (v[i] < other[i])
                v[i] = other[i];
        }
    }

    inline unsigned int index(const vec<D,T>& size) const {
        unsigned int multiplier = 1;
        unsigned int ret = 0;
        FOR_EACH_DIM(i) {
            ret += v[D - i - 1] * multiplier;
            multiplier *= size[D - i - 1];
        }
        return ret;
    }

    inline bool index_inc(const vec<D,T>& size) {
        FOR_EACH_DIM(i) {
            v[D - i - 1] += 1;
            if (v[D - i - 1] < size[D - i - 1])
                return true;
            else
                v[D - i - 1] = 0;
        }

        return false;
    }

    inline T inner(const vec<D,T>& other) const {
        T ret = 0;
        FOR_EACH_DIM(d)
            ret += v[d]*other[d];
        return ret;
    }


    /***************************************************************************
     * Compare operators
     **************************************************************************/

    inline bool less(const vec<D,T>& upper, unsigned int dim) const {
        return v[dim] < upper[dim];
    }

    inline bool less(const vec<D,T>& upper) const {
        FOR_EACH_DIM(i) {
            if (!(v[i] < upper[i]))
                return false;
        }
        return true;
    }

    inline bool lessEq(const vec<D,T>& upper) const {
        FOR_EACH_DIM(i) {
            if (v[i] > upper[i])
                return false;
        }
        return true;
    }

    inline bool moreEq(const vec<D,T>& lower) const {
        FOR_EACH_DIM(i) {
            if (v[i] < lower[i])
                return false;
        }
        return true;
    }

    inline double squareDist(const vec<D,T>& other) const {
        double ret = 0;
        FOR_EACH_DIM(i) {
            double diff = v[i] - other[i];
            ret += diff*diff;
        }
        return ret;
    }

    /***************************************************************************
     * Properties operators
     **************************************************************************/

    inline double L1Scalar() const {
        double ret = 0;
        FOR_EACH_DIM(i)
            ret += v[i];
        return ret;
    }

    inline double squareScalar() const {
        double ret = 0;
        FOR_EACH_DIM(i)
            ret += (double)(v[i])*(double)(v[i]);
        return ret;
    }

    inline double scalar() const {
        double ret = this->squareScalar();
        return std::sqrt(ret);
    }

    inline double area() const {
        double ret = 1;
        FOR_EACH_DIM(i)
            ret *= v[i];
        return ret;
    }
};


#define FOR_EACH_INDEX(__index__, __limit__)	\
    (__index__).reset(); 						\
    for (bool __cont__##__index__=true; 		\
        __cont__##__index__;					\
        __cont__##__index__=(__index__).index_inc(__limit__))


template<typename VEC_TYPE1, typename VEC_TYPE2, typename VEC_TYPE3>
inline void vec_add(const VEC_TYPE1& a, const VEC_TYPE2& b, VEC_TYPE3& res) {
    FOR_EACH_DIM_D(i, a.dim)
        res[i] = a[i] + b[i];
}


template<typename VEC_TYPE1, typename VEC_TYPE2, typename VEC_TYPE3>
inline void vec_dec(const VEC_TYPE1& a, const VEC_TYPE2& b, VEC_TYPE3& res) {
    FOR_EACH_DIM_D(i, a.dim)
        res[i] = a[i] - b[i];
}

#endif /* VEC_HPP_ */
