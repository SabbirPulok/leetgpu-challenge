#pragma once

#include <bell_format.hpp>
#include <sell_format.hpp>
#include <sparse_format.hpp>

template<typename T>
void launch_spgemm_kernel(const T& matrix_a, const T& matrix_b, T& matrix_c);