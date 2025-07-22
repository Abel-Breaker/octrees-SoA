#pragma once

#include "point_encoder.hpp"

namespace PointEncoding {

class HilbertEncoder3D : public PointEncoder {
public:
    /// @brief The maximum depth that this encoding allows (in Hilbert 64 bit integers, we need 3 bits for each level, so 21)
    static constexpr uint32_t MAX_DEPTH = 21;

    /// @brief The minimum unit of length of the encoded coordinates
    static constexpr double EPS = 1.0f / (1 << MAX_DEPTH);

    /// @brief The minimum (strict) upper bound for every Hilbert code. Equal to the unused bit followed by 63 zeros.
    static constexpr key_t UPPER_BOUND = 0x8000000000000000;

    /// @brief The amount of bits that are not used, in Hilbert encodings this is the MSB of the key
    static constexpr uint32_t UNUSED_BITS = 1;

    /// @brief A constant array to map adequately rotated x, y, z coordinates to their corresponding octant 
    static constexpr coords_t mortonToHilbert[8] = {0, 1, 3, 2, 7, 6, 4, 5};

    /**
     * @brief Encodes the given integer coordinates in the range [0,2^MAX_DEPTH]x[0,2^MAX_DEPTH]x[0,2^MAX_DEPTH] into their Hilbert key
     * The algorithm is described in the citations above but consists on something similar to the intertwinement of bits of Morton codes 
     * but with some extra rotations and reflections in each step.
     */
    key_t encode(coords_t x, coords_t y, coords_t z) const override {
        key_t key = 0;

        for(int level = MAX_DEPTH - 1; level >= 0; level--) {
            // Find octant and append to the key (same as Morton codes)
            const coords_t xi = (x >> level) & 1u;
            const coords_t yi = (y >> level) & 1u;
            const coords_t zi = (z >> level) & 1u;

            const coords_t octant = (xi << 2) | (yi << 1) | zi;
            key <<= 3;
            key |= mortonToHilbert[octant];
            
            // Turn x, y, z (Karnaugh mapped operations, check citation and Lam and Shapiro paper detailing 2D case 
            // for understanding how this works)
            x ^= -(xi & ((!yi) | zi));
            y ^= -((xi & (yi | zi)) | (yi & (!zi)));
            z ^= -((xi & (!yi) & (!zi)) | (yi & (!zi)));

            if (zi) {
                // Cylic anticlockwise rotation x, y, z -> y, z, x
                coords_t temp = x;
                x = y, y = z, z = temp;
            } else if (!yi) {
                // Swap x and z
                coords_t temp = x;
                x = z, z = temp;
            }
        }

        return key;
    }
    /**
     * @brief Encodes the given integer coordinates in the range [0,2^MAX_DEPTH]x[0,2^MAX_DEPTH]x[0,2^MAX_DEPTH] into their Hilbert key
     * The algorithm is described in the citations above but consists on something similar to the intertwinement of bits of Morton codes 
     * but with some extra rotations and reflections in each step.
     */
    void encodeVectorized(const uint32_t *x, const uint32_t *y, const uint32_t *z, std::vector<key_t> &keys, size_t i) const override
    {

        // Inicializar claves de 64 bits como 2 vectores separados
        __m256i key_lo = _mm256_setzero_si256(); // para elementos 0–3
        __m256i key_hi = _mm256_setzero_si256(); // para elementos 4–7

        __m256i vx = _mm256_loadu_si256((__m256i *)(const uint32_t *)x);
        __m256i vy = _mm256_loadu_si256((__m256i *)(const uint32_t *)y);
        __m256i vz = _mm256_loadu_si256((__m256i *)(const uint32_t *)z);

        __m256i one = _mm256_set1_epi32(1);

        alignas(32) uint32_t zi_arr[8], yi_arr[8]; // For rotations

        for (int level = MAX_DEPTH - 1; level >= 0; level--)
        {
            // Find octant and append to the key (same as Morton codes)
            __m256i shift = _mm256_set1_epi32(level);

            __m256i xi = _mm256_and_si256(_mm256_srlv_epi32(vx, shift), one);
            __m256i yi = _mm256_and_si256(_mm256_srlv_epi32(vy, shift), one);
            __m256i zi = _mm256_and_si256(_mm256_srlv_epi32(vz, shift), one);

            __m256i octant = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(xi, 2), _mm256_slli_epi32(yi, 1)), zi);

            // Lookup mortonToHilbert[octant]
            alignas(32) uint32_t octants[8];
            _mm256_store_si256((__m256i *)octants, octant);

            alignas(32) uint32_t mthVals[8];
            for (int j = 0; j < 8; ++j)
            {
                mthVals[j] = mortonToHilbert[octants[j]];
            }
 
            __m256i hilbertVals = _mm256_load_si256((__m256i *)mthVals);

            // Expandir Morton→Hilbert (8x uint32_t) a 8x uint64_t
            __m128i mth_lo = _mm256_extracti128_si256(hilbertVals, 0);
            __m128i mth_hi = _mm256_extracti128_si256(hilbertVals, 1);
            __m256i mth64_lo = _mm256_cvtepu32_epi64(mth_lo); // 4x uint64_t
            __m256i mth64_hi = _mm256_cvtepu32_epi64(mth_hi); // 4x uint64_t

            // Shift claves y combinarlas
            key_lo = _mm256_slli_epi64(key_lo, 3);
            key_hi = _mm256_slli_epi64(key_hi, 3);
            key_lo = _mm256_or_si256(key_lo, mth64_lo);
            key_hi = _mm256_or_si256(key_hi, mth64_hi);


            // Turn x, y, z (Karnaugh mapped operations, check citation and Lam and Shapiro paper detailing 2D case
            // for understanding how this works)
            // Apply Karnaugh-style bit transformations
            __m256i not_yi = _mm256_xor_si256(yi, one);
            __m256i not_zi = _mm256_xor_si256(zi, one);

            __m256i cond_x = _mm256_and_si256(xi, _mm256_or_si256(not_yi, zi));
            __m256i mask_x = _mm256_sub_epi32(_mm256_setzero_si256(), cond_x);
            vx = _mm256_xor_si256(vx, mask_x);

            __m256i cond_y = _mm256_or_si256(_mm256_and_si256(xi, _mm256_or_si256(yi, zi)),
                                             _mm256_and_si256(yi, not_zi));
            __m256i mask_y = _mm256_sub_epi32(_mm256_setzero_si256(), cond_y);
            vy = _mm256_xor_si256(vy, mask_y);

            __m256i cond_z = _mm256_or_si256(_mm256_and_si256(xi, _mm256_and_si256(not_yi, not_zi)),
                                             _mm256_and_si256(yi, not_zi));
            __m256i mask_z = _mm256_sub_epi32(_mm256_setzero_si256(), cond_z);
            vz = _mm256_xor_si256(vz, mask_z);

            // Rotation or swap
            _mm256_store_si256((__m256i *)zi_arr, zi);
            _mm256_store_si256((__m256i *)yi_arr, yi);

            alignas(32) uint32_t tx[8], ty[8], tz[8];
            _mm256_store_si256((__m256i *)tx, vx);
            _mm256_store_si256((__m256i *)ty, vy);
            _mm256_store_si256((__m256i *)tz, vz);

            for (int j = 0; j < 8; ++j)
            {
                if (zi_arr[j])
                {
                    uint32_t tmp = tx[j];
                    tx[j] = ty[j];
                    ty[j] = tz[j];
                    tz[j] = tmp;
                }
                else if (!yi_arr[j])
                {
                    std::swap(tx[j], tz[j]);
                }
            }

            vx = _mm256_load_si256((__m256i *)tx);
            vy = _mm256_load_si256((__m256i *)ty);
            vz = _mm256_load_si256((__m256i *)tz);
        }

        // Store the final key
        _mm256_storeu_si256((__m256i *)&keys[i], key_lo);
        _mm256_storeu_si256((__m256i *)&keys[i+4], key_hi);

    }

    /// @brief Decodes the given key and puts the coordinates into x, y, z
    void decode(key_t code, coords_t &x, coords_t &y, coords_t &z) const override {
        // Initialize the coords values
        x = 0, y = 0, z = 0;
        for(int level = 0; level < MAX_DEPTH; level++) {
            // Extract the octant from the key and put the bits into xi, yi and zi
            const coords_t octant   = (code >> (3 * level)) & 7u;
            const coords_t xi = octant >> 2u;
            const coords_t yi = (octant >> 1u) & 1u;
            const coords_t zi = octant & 1u;

            if(yi ^ zi) {
                // Cylic clockwise rotation x, y, z -> z, x, y
                coords_t temp = x;
                x = z, z = y, y = temp;
            } else if((!xi & !yi & !zi) || (xi & yi & zi)) {
                // Swap x and z
                coords_t temp = x;
                x = z, z = temp;
            }

            // Turn x, y, z (Karnaugh mapped operations, check citation and Lam and Shapiro paper detailing 2D case 
            // for understanding how this works)
            const coords_t mask = (1u << level) - 1u;
            x ^= mask & (-(xi & (yi | zi)));
            y ^= mask & (-((xi & ((!yi) | (!zi))) | ((!xi) & yi & zi)));
            z ^= mask & (-((xi & (!yi) & (!zi)) | (yi & zi)));

            // Append the new bit to the position
            x |= (xi << level);
            y |= ((xi ^ yi) << level);
            z |= ((yi ^ zi) << level);
        }
        return;
    }

    key_t encodeFromPoint(const Point& p, const Box &bbox) const override {
        coords_t x, y, z;
        getAnchorCoords(p, bbox, x, y, z);
		return encode(x, y, z);
    }

    // Getters
    inline uint32_t maxDepth() const override { return MAX_DEPTH; }
    inline double eps() const override { return EPS; }
    inline key_t upperBound() const override { return UPPER_BOUND; }
    inline uint32_t unusedBits() const override { return UNUSED_BITS; }
    inline std::string getEncoderName() const override { return "HilbertEncoder3D"; };
    inline std::string getShortEncoderName() const override { return "hilb"; };
};

} // namespace PointEncoding
