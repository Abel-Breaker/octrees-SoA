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
    static constexpr uint32_t mortonToHilbert[8] = {0, 1, 3, 2, 7, 6, 4, 5};

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
        __m256i zero = _mm256_setzero_si256();

        __m256i lookup_table_mortonToHilbert = _mm256_loadu_si256((const __m256i*)mortonToHilbert);

        alignas(32) uint32_t zi_arr[8], yi_arr[8]; // For rotations

        for (int level = MAX_DEPTH - 1; level >= 0; level--)
        {
            // Find octant and append to the key (same as Morton codes)
            __m256i shift = _mm256_set1_epi32(level);

            __m256i xi = _mm256_and_si256(_mm256_srlv_epi32(vx, shift), one);
            __m256i yi = _mm256_and_si256(_mm256_srlv_epi32(vy, shift), one);
            __m256i zi = _mm256_and_si256(_mm256_srlv_epi32(vz, shift), one);

            __m256i octant = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(xi, 2), _mm256_slli_epi32(yi, 1)), zi);

            __m256i hilbertVals = _mm256_permutevar8x32_epi32(lookup_table_mortonToHilbert, octant);

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
            __m256i mask_x = _mm256_sub_epi32(zero, cond_x);
            vx = _mm256_xor_si256(vx, mask_x);

            __m256i cond_y = _mm256_or_si256(_mm256_and_si256(xi, _mm256_or_si256(yi, zi)),
                                             _mm256_and_si256(yi, not_zi));
            __m256i mask_y = _mm256_sub_epi32(zero, cond_y);
            vy = _mm256_xor_si256(vy, mask_y);

            __m256i cond_z = _mm256_or_si256(_mm256_and_si256(xi, _mm256_and_si256(not_yi, not_zi)),
                                             _mm256_and_si256(yi, not_zi));
            __m256i mask_z = _mm256_sub_epi32(zero, cond_z);
            vz = _mm256_xor_si256(vz, mask_z);

            // rot_mask = zi ? 0xFFFFFFFF : 0x00000000
            __m256i rot_mask = _mm256_cmpeq_epi32(zi, one);

            // swap_mask = (!zi && !yi) ? 0xFFFFFFFF : 0x00000000
            __m256i swap_mask = _mm256_and_si256(
                _mm256_cmpeq_epi32(zi, zero),
                _mm256_cmpeq_epi32(yi, zero));

            // Paso 2: Rotación condicional: tx = ty, ty = tz, tz = tx (original) cuando zi == 1
            __m256i tx_orig = vx;
            vx = _mm256_blendv_epi8(vx, vy, rot_mask);      // tx = ty
            vy = _mm256_blendv_epi8(vy, vz, rot_mask);      // ty = tz
            vz = _mm256_blendv_epi8(vz, tx_orig, rot_mask); // tz = tx original


            // Paso 3: Swap condicional entre tx y tz cuando (!zi && !yi)
            __m256i tmp_vx = _mm256_blendv_epi8(vx, vz, swap_mask); // tmp = vz where mask
            __m256i tmp_vz = _mm256_blendv_epi8(vz, vx, swap_mask); // vz = vx where mask
            vx = tmp_vx;
            vz = tmp_vz; 
        }

        // Store the final key
        _mm256_storeu_si256((__m256i *)&keys[i], key_lo);
        _mm256_storeu_si256((__m256i *)&keys[i+4], key_hi);

    }

    void encodeVectorizedAVX512(const uint32_t *x, const uint32_t *y, const uint32_t *z, std::vector<key_t> &keys, size_t i) const override
{
    __m512i vx = _mm512_load_si512((__m512i *)x);
    __m512i vy = _mm512_load_si512((__m512i *)y);
    __m512i vz = _mm512_load_si512((__m512i *)z);

    __m512i one = _mm512_set1_epi32(1);

    // A constant array duplicated (for avx use) to map adequately rotated x, y, z coordinates to their corresponding octant 
    __m512i lookup_table_mortonToHilbert = _mm512_setr_epi32(
        0, 1, 3, 2, 7, 6, 4, 5, // Only use this
        0, 1, 3, 2, 7, 6, 4, 5 // Duplicated for AVX512 (not used)
    );

    __m512i key_lo = _mm512_setzero_si512();
    __m512i key_hi = _mm512_setzero_si512();


    for (int level = MAX_DEPTH - 1; level >= 0; --level)
    {
        __m512i shift = _mm512_set1_epi32(level);

        // xi, yi, zi = (coord >> shift) & 1
        __m512i xi = _mm512_and_epi32(_mm512_srlv_epi32(vx, shift), one);
        __m512i yi = _mm512_and_epi32(_mm512_srlv_epi32(vy, shift), one);
        __m512i zi = _mm512_and_epi32(_mm512_srlv_epi32(vz, shift), one);

        
        __m512i xi2 = _mm512_slli_epi32(xi, 2);
        __m512i yi1 = _mm512_slli_epi32(yi, 1);
        __m512i octant = _mm512_or_epi32(_mm512_or_epi32(xi2, yi1), zi);

        // Utilice octant indexs (16 x [0-7]) to acces the first part of the lookup table
        __m512i hilbertVals = _mm512_permutexvar_epi32(octant, lookup_table_mortonToHilbert);

        // Convertir hilbertVals (16 x uint32_t) → dos vectores de 8 x uint64_t
        __m256i lo = _mm512_castsi512_si256(hilbertVals);       // primeros 8
        __m256i hi = _mm512_extracti32x8_epi32(hilbertVals, 1); // últimos 8

        __m512i hilbertVals_lo = _mm512_cvtepu32_epi64(lo); // 8 x uint64_t
        __m512i hilbertVals_hi = _mm512_cvtepu32_epi64(hi); // 8 x uint64_t

        // Desplazar las claves actuales (64-bit) 3 bits a la izquierda
        key_lo = _mm512_slli_epi64(key_lo, 3);
        key_hi = _mm512_slli_epi64(key_hi, 3);

        // Combinar (OR) con los valores convertidos
        key_lo = _mm512_or_si512(key_lo, hilbertVals_lo);
        key_hi = _mm512_or_si512(key_hi, hilbertVals_hi);

        // === Bit manipulation: Karnaugh-style operations ===

        // Recompute masks
        __m512i not_yi = _mm512_xor_epi32(yi, one);
        __m512i not_zi = _mm512_xor_epi32(zi, one);

        // X
        __m512i cond_x = _mm512_and_epi32(xi, _mm512_or_epi32(not_yi, zi));
        vx = _mm512_xor_epi32(vx, _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(cond_x, one), -1));

        // Y
        __m512i cond_y = _mm512_or_epi32(
            _mm512_and_epi32(xi, _mm512_or_epi32(yi, zi)),
            _mm512_and_epi32(yi, not_zi)
        );
        vy = _mm512_xor_epi32(vy, _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(cond_y, one), -1));

        // Z
        __m512i cond_z = _mm512_or_epi32(
            _mm512_and_epi32(xi, _mm512_and_epi32(not_yi, not_zi)),
            _mm512_and_epi32(yi, not_zi));
        vz = _mm512_xor_epi32(vz, _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(cond_z, one), -1));

        // Creamos máscaras (cada bit representa una condición para un elemento)
        __mmask16 mask_octants = _mm512_test_epi32_mask(zi, _mm512_set1_epi32(-1));
        __mmask16 mask_mthVals_false = _mm512_testn_epi32_mask(yi, _mm512_set1_epi32(-1));
        __mmask16 mask_else = ~mask_octants & mask_mthVals_false;

        // Rotación para `octants == true`: tx = ty, ty = tz, tz = tx_original
        __m512i vx_orig = vx;
        vx = _mm512_mask_mov_epi32(vx, mask_octants, vy);
        vy = _mm512_mask_mov_epi32(vy, mask_octants, vz);
        vz = _mm512_mask_mov_epi32(vz, mask_octants, vx_orig);

        // Swap tx <-> tz para el caso else
        __m512i tmp_vx = vx;
        vx = _mm512_mask_mov_epi32(vx, mask_else, vz);
        vz = _mm512_mask_mov_epi32(vz, mask_else, tmp_vx);
    }

    // Store final key (two 256-bit stores for 8x uint64_t keys)
    _mm512_storeu_si512((__m512i *)&keys[i], key_lo);
    _mm512_storeu_si512((__m512i *)&keys[i + 8], key_hi);
}
/*void encodeVectorizedAVX512(const uint32_t *x, const uint32_t *y, const uint32_t *z, std::vector<key_t> &keys, size_t i) const override
{
    __m512i vx = _mm512_loadu_si512((__m512i *)x);
    __m512i vy = _mm512_loadu_si512((__m512i *)y);
    __m512i vz = _mm512_loadu_si512((__m512i *)z);

    __m512i one = _mm512_set1_epi32(1);

    __m512i key_lo = _mm512_setzero_si512();
    __m512i key_hi = _mm512_setzero_si512();

    alignas(64) uint32_t tx[16], ty[16], tz[16];
    alignas(64) uint32_t octants[16];
    alignas(64) uint32_t mthVals[16];


    for (int level = MAX_DEPTH - 1; level >= 0; --level)
    {
        __m512i shift = _mm512_set1_epi32(level);

        // xi, yi, zi = (coord >> shift) & 1
        __m512i xi = _mm512_and_epi32(_mm512_srlv_epi32(vx, shift), one);
        __m512i yi = _mm512_and_epi32(_mm512_srlv_epi32(vy, shift), one);
        __m512i zi = _mm512_and_epi32(_mm512_srlv_epi32(vz, shift), one);

        __m512i xi2 = _mm512_slli_epi32(xi, 2);
        __m512i yi1 = _mm512_slli_epi32(yi, 1);
        __m512i octant = _mm512_or_epi32(_mm512_or_epi32(xi2, yi1), zi);

        // Store octant to array
        _mm512_store_si512((__m512i *)octants, octant);

        // Lookup mortonToHilbert
        for (int j = 0; j < 16; ++j)
            mthVals[j] = mortonToHilbert[octants[j]];

        __m512i hilbertVals = _mm512_load_si512((__m512i *)mthVals);

        // Convertir hilbertVals (16 x uint32_t) → dos vectores de 8 x uint64_t
        __m256i lo = _mm512_castsi512_si256(hilbertVals);       // primeros 8
        __m256i hi = _mm512_extracti32x8_epi32(hilbertVals, 1); // últimos 8

        __m512i hilbertVals_lo = _mm512_cvtepu32_epi64(lo); // 8 x uint64_t
        __m512i hilbertVals_hi = _mm512_cvtepu32_epi64(hi); // 8 x uint64_t

        // Desplazar las claves actuales (64-bit) 3 bits a la izquierda
        key_lo = _mm512_slli_epi64(key_lo, 3);
        key_hi = _mm512_slli_epi64(key_hi, 3);

        // Combinar (OR) con los valores convertidos
        key_lo = _mm512_or_si512(key_lo, hilbertVals_lo);
        key_hi = _mm512_or_si512(key_hi, hilbertVals_hi);

        // === Bit manipulation: Karnaugh-style operations ===

        // Recompute masks
        __m512i not_yi = _mm512_xor_epi32(yi, one);
        __m512i not_zi = _mm512_xor_epi32(zi, one);

        // X
        __m512i cond_x = _mm512_and_epi32(xi, _mm512_or_epi32(not_yi, zi));
        vx = _mm512_xor_epi32(vx, _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(cond_x, one), -1));

        // Y
        __m512i cond_y = _mm512_or_epi32(
            _mm512_and_epi32(xi, _mm512_or_epi32(yi, zi)),
            _mm512_and_epi32(yi, not_zi)
        );
        vy = _mm512_xor_epi32(vy, _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(cond_y, one), -1));

        // Z
        __m512i cond_z = _mm512_or_epi32(
            _mm512_and_epi32(xi, _mm512_and_epi32(not_yi, not_zi)),
            _mm512_and_epi32(yi, not_zi)
        );
        vz = _mm512_xor_epi32(vz, _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(cond_z, one), -1));

        // Store for scalar rotation
        _mm512_store_si512((__m512i *)tx, vx);
        _mm512_store_si512((__m512i *)ty, vy);
        _mm512_store_si512((__m512i *)tz, vz);
        _mm512_store_si512((__m512i *)octants, zi); // reusing octants buffer for zi
        _mm512_store_si512((__m512i *)mthVals, yi); // reusing mthVals buffer for yi

        for (int j = 0; j < 16; ++j)
        {
            if (octants[j])
            {
                uint32_t tmp = tx[j];
                tx[j] = ty[j];
                ty[j] = tz[j];
                tz[j] = tmp;
            }
            else if (!mthVals[j])
            {
                std::swap(tx[j], tz[j]);
            }
        }

        // Reload rotated vectors
        vx = _mm512_load_si512((__m512i *)tx);
        vy = _mm512_load_si512((__m512i *)ty);
        vz = _mm512_load_si512((__m512i *)tz);
    }

    // Store final key (two 256-bit stores for 8x uint64_t keys)
    _mm512_storeu_si512((__m512i *)&keys[i], key_lo);
    _mm512_storeu_si512((__m512i *)&keys[i+8], key_hi);
}*/

/// @brief Decodes the given key and puts the coordinates into x, y, z
void decode(key_t code, coords_t &x, coords_t &y, coords_t &z) const override
{
    // Initialize the coords values
    x = 0, y = 0, z = 0;
    for (int level = 0; level < MAX_DEPTH; level++)
    {
        // Extract the octant from the key and put the bits into xi, yi and zi
        const coords_t octant = (code >> (3 * level)) & 7u;
        const coords_t xi = octant >> 2u;
        const coords_t yi = (octant >> 1u) & 1u;
        const coords_t zi = octant & 1u;

        if (yi ^ zi)
        {
            // Cylic clockwise rotation x, y, z -> z, x, y
            coords_t temp = x;
            x = z, z = y, y = temp;
        }
        else if ((!xi & !yi & !zi) || (xi & yi & zi))
        {
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
