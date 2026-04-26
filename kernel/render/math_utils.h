#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdio.h>
#include <iostream>

constexpr float InvPi = 0.31830988618379067154;
constexpr float PIOver4 = 0.78539816339744830961;
constexpr float PIOver2 = 1.57079632679489661923;

template <typename T>
class vector3 {
public:
	T x, y, z;

	__host__ __device__ T length() const
	{
		return std::sqrt(x * x + y * y + z * z);
	}

	__host__ __device__ vector3() :x(0), y(0), z(0) {}
	__host__ __device__ vector3(T x, T y, T z) : x(x), y(y), z(z) {}

	__host__ __device__ vector3<T> operator+(const vector3<T>& v) const {
		return vector3(x + v.x, y + v.y, z + v.z);
	}

	__host__ __device__ vector3<T>& operator+=(const vector3<T>& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	__host__ __device__ vector3<T> operator*(const vector3<T>& v) const {
		return vector3(x * v.x, y * v.y, z * v.z);
	}

	__host__ __device__ vector3<T>& operator*=(const vector3<T>& v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}

	__host__ __device__ vector3<T> operator-() const {

		return vector3(-x, -y, -z);
	}

	__host__ __device__ vector3<T> operator-(const vector3<T>& v) const {
		return vector3(x - v.x, y - v.y, z - v.z);
	}

	__host__ __device__ vector3<T>& operator-=(const vector3<T>& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	__host__ __device__ vector3<T>& operator*=(T a) {
		x *= a;
		y *= a;
		z *= a;
		return *this;
	}

	__host__ __device__ vector3<T> operator*(T a) const {

		return vector3(a * x, a * y, a * z);
	}

	__host__ __device__ vector3<T> normalize() const
	{
		T len = length();
		if (len == 0)
		{
			return *this;
		}
		else
		{
			return (*this) * (1.0 / len);
		}
	}
};
template <typename T>
__host__ __device__ vector3<T> operator*(T a, const vector3<T>& v)
{
	return v * a;
};

template <typename T>
__host__ __device__ vector3<T> cross(const vector3<T>& a, const vector3<T>& b)
{
	return vector3<T>(a.y * b.z - b.y * a.z, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
};


template <typename T>
__host__ __device__ T dot(const vector3<T>& a, const vector3<T>& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
};

template <typename T>
class vector2 {
public:
	T x, y;
	__host__ __device__ vector2() :x(0), y(0) {}
	__host__ __device__ vector2(T x, T y) : x(x), y(y) {}

	__host__ __device__ vector2<T> operator+(const vector2<T>& v) const {
		return vector2(x + v.x, y + v.y);
	}

	__host__ __device__ vector2<T>& operator+=(const vector2<T>& v) {
		x += v.x;
		y += v.y;
		return *this;
	}

	__host__ __device__ vector2<T> operator-(const vector2<T>& v) const {
		return vector2(x - v.x, y - v.y);
	}

	__host__ __device__ vector2<T>& operator-=(const vector2<T>& v) {
		x -= v.x;
		y -= v.y;
		return *this;
	}

	__host__ __device__ vector2<T>& operator*=(T a) {
		x *= a;
		y *= a;
		return *this;
	}

	__host__ __device__ vector2<T> operator*(T a) const {

		return vector2(a * x, a * y);
	}
};
template <typename T>
__host__ __device__ vector2<T> operator*(T a, const vector2<T>& v)
{
	return v * a;
};


typedef vector3<float> vec3f;
typedef vector2<float> vec2f;

template <typename T,typename U, typename V>
constexpr T Clamp(T val, U low, V high)
{
	if (val < low) return T(low);
	else if (val > high) return T(high);
	else return val;

};

template <typename Predicate>
size_t FindInterval(size_t N, const Predicate& pred)
{
	using ssize_t = std::make_signed_t<size_t>;
	ssize_t size = (ssize_t)N - 2, first = 1;
	while (size > 0)
	{
		size_t half = (size_t)size >> 1, middle = first + half;
		bool predResult = pred(middle);
		first = predResult ? middle + 1 : first;
		size = predResult ? size - (half + 1) : half;
	}
	return (size_t)Clamp((ssize_t)first - 1, 0, N - 2);
};

__device__ inline float Interpolate(float t, float a, float b)
{
	return t * a + (1 - t) * b;
};



__device__ inline vec2f SampleUniformDiskConcentric(vec2f u)
{
	vec2f offset = 2.0f * u - vec2f(1, 1);
	if (offset.x == 0 && offset.y == 0)
	{
		return vec2f(0, 0);
	}
	float theta, r;
	if (std::abs(offset.x) > std::abs(offset.y))
	{
		r = offset.x;
		theta = PIOver4 * (offset.y / offset.x);
	}
	else
	{
		r = offset.y;
		theta = PIOver2 - PIOver4 * (offset.x / offset.y);
	}
	return r * vec2f(std::cos(theta), std::sin(theta));
};


__device__ inline vec3f SampleCosineHemiSphere(vec2f u)
{
	vec2f d = SampleUniformDiskConcentric(u);
	float z = 1.0f - d.x * d.x - d.y * d.y;
	if (z >= 0.0f)
	{
		z = std::sqrt(z);
	}
	else
	{
		z = 0.0f;
	}
	return vec3f(d.x, d.y, z);
};

__device__ inline float CosineHemiSpherePDF(float costheta)
{
	return costheta * InvPi;
}

__device__ inline float AbsCosineTheta(vec3f w)
{
	return std::abs(w.z);
}


class Mat3
{
public:
	Mat3(float m00=0, float m01=0, float m02=0,
		float m10=0, float m11=0, float m12=0,
		float m20=0, float m21=0, float m22=0)
	{
		data[0][0] = m00; data[0][1] = m01; data[0][2] = m02;
		data[1][0] = m10; data[1][1] = m11; data[1][2] = m12;
		data[2][0] = m20; data[2][1] = m21; data[2][2] = m22;
	}

	Mat3 inverse() const
	{
		float m00 = data[0][0], m01 = data[0][1], m02 = data[0][2];
		float m10 = data[1][0], m11 = data[1][1], m12 = data[1][2];
		float m20 = data[2][0], m21 = data[2][1], m22 = data[2][2];

		// 计算行列式
		float det = m00 * (m11 * m22 - m12 * m21)
			- m01 * (m10 * m22 - m12 * m20)
			+ m02 * (m10 * m21 - m11 * m20);

		
		if (fabs(det) < 1e-6f) {
			return Mat3();
		}

		float inv_det = 1.0f / det;

		
		Mat3 result(
			
			(m11 * m22 - m12 * m21) * inv_det,
			(m02 * m21 - m01 * m22) * inv_det,
			(m01 * m12 - m02 * m11) * inv_det,
			
			(m12 * m20 - m10 * m22) * inv_det,
			(m00 * m22 - m02 * m20) * inv_det,
			(m02 * m10 - m00 * m12) * inv_det,
			
			(m10 * m21 - m11 * m20) * inv_det,
			(m01 * m20 - m00 * m21) * inv_det,
			(m00 * m11 - m01 * m10) * inv_det
		);

		return result;
	}

	static Mat3 Diag(float d0, float d1, float d2)
	{
		return Mat3(
			d0, 0.0f, 0.0f,
			0.0f, d1, 0.0f,
			0.0f, 0.0f, d2
		);
	}

	vec3f operator*(const vec3f &b) const
	{
		vec3f res;
		res.x = data[0][0] * b.x + data[0][1] * b.y + data[0][2] * b.z;
		res.y = data[1][0] * b.x + data[1][1] * b.y + data[1][2] * b.z;
		res.z = data[2][0] * b.x + data[2][1] * b.y + data[2][2] * b.z;

		return res;
	}

	Mat3 operator*(const Mat3& b) const
	{
		Mat3 res;
		const auto& a = *this; 

		
		res.data[0][0] = a.data[0][0] * b.data[0][0] + a.data[0][1] * b.data[1][0] + a.data[0][2] * b.data[2][0];
		res.data[0][1] = a.data[0][0] * b.data[0][1] + a.data[0][1] * b.data[1][1] + a.data[0][2] * b.data[2][1];
		res.data[0][2] = a.data[0][0] * b.data[0][2] + a.data[0][1] * b.data[1][2] + a.data[0][2] * b.data[2][2];

		res.data[1][0] = a.data[1][0] * b.data[0][0] + a.data[1][1] * b.data[1][0] + a.data[1][2] * b.data[2][0];
		res.data[1][1] = a.data[1][0] * b.data[0][1] + a.data[1][1] * b.data[1][1] + a.data[1][2] * b.data[2][1];
		res.data[1][2] = a.data[1][0] * b.data[0][2] + a.data[1][1] * b.data[1][2] + a.data[1][2] * b.data[2][2];

		res.data[2][0] = a.data[2][0] * b.data[0][0] + a.data[2][1] * b.data[1][0] + a.data[2][2] * b.data[2][0];
		res.data[2][1] = a.data[2][0] * b.data[0][1] + a.data[2][1] * b.data[1][1] + a.data[2][2] * b.data[2][1];
		res.data[2][2] = a.data[2][0] * b.data[0][2] + a.data[2][1] * b.data[1][2] + a.data[2][2] * b.data[2][2];

		return res;
	}


private:
	float data[3][3];
};