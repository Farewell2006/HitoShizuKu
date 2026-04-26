#pragma once
#include <math_utils.h>


class XYZ
{
public:
	XYZ(float x, float y, float z) :x(x), y(y), z(z) {}
	float average()const { return (x + y + z) / 3; }

	vec2f xy() const {
		return vec2f(x / (x + y + z), y / (x + y + z));

	}

	static XYZ FromxyY(vec2f xy, float Y = 1.0f)
	{
		if (xy.y == 0)
		{
			return XYZ(0, 0, 0);
		}
		return XYZ(xy.y * Y / xy.y, Y, (1 - xy.x - xy.y) * Y / xy.y);
	}

	XYZ& operator+=(const XYZ& a)
	{
		x += a.x;
		y += a.y;
		z += a.z;
		return *this;
	}

	XYZ& operator-=(const XYZ& a)
	{
		x -= a.x;
		y -= a.y;
		z -= a.z;
		return *this;
	}

	XYZ operator+(const XYZ& a)const {
		XYZ res = *this;
		return res += a;
	}

	XYZ operator-(const XYZ& a)const {
		XYZ res = *this;
		return res -= a;
	}

	friend XYZ operator-(const float a, const XYZ& b)
	{
		return XYZ(a - b.x, a - b.y, a - b.z);
	}

	XYZ& operator*=(const XYZ& b)
	{
		x *= b.x;
		y *= b.y;
		z *= b.z;
		return *this;
	}

	XYZ operator*(const XYZ& b)
	{
		XYZ res = *this;
		return this *= b;
	}

	XYZ operator*(float a)
	{
		XYZ res = *this;
		return XYZ(a * res.x, a * res.y, a * res.z);
	}

	XYZ& operator*=(float a)
	{
		x *= a;
		y *= a;
		z *= a;
		return *this;
	}

	XYZ& operator/=(const XYZ& b)
	{
		x /= b.x;
		y /= b.y;
		z /= b.z;
		return *this;
	}

	XYZ operator/(const XYZ& b)
	{
		XYZ res = *this;
		return res /= b;
	}

	XYZ& operator/=(float a)
	{
		//如果a为0 什么都不做
		if (a != 0.0f)
		{
			x /= a;
			y /= a;
			z /= a;
		}
		return *this;
	}

	XYZ operator/(float a)
	{
		XYZ res = *this;
		return res /= a;
	}

	XYZ operator-()const {
		return { -x,-y,-z };
	}

	bool operator==(const XYZ& b) const { return x == b.x && y == b.y && z == b.z; }
	bool operator!=(const XYZ& b) const { return x != b.x || y != b.y || z != b.z; }

	float operator[](int i)const {
		if (i == 0)
		{
			return x;
		}
		else if (i == 1)
		{
			return y;
		}
		else if (i == 2)
		{
			return z;
		}
		else
		{
			return 0.0f;
		}
	}

	float& operator[](int i)
	{
		if (i == 0)
		{
			return x;
		}
		else if (i == 1)
		{
			return y;
		}
		else if (i == 2)
		{
			return z;
		}
		else
		{
			return 0.0f;
		}
	}

	float x = 0.0f, y = 0.0f, z = 0.0f;
};

class RGB
{
public:
	float r = 0.0f, g = 0.0f, b = 0.0f;
	RGB(float r, float g, float b) :r(r), g(g), b(b) {}

	RGB& operator+=(RGB x)
	{
		r += x.r;
		g += x.g;
		b += x.b;
		return *this;
	}

	RGB operator+(RGB x)const
	{
		RGB res = *this;
		return res += x;
	}

	RGB& operator-=(RGB x)
	{
		r -= x.r;
		g -= x.g;
		b -= x.b;
		return *this;
	}

	RGB operator-(RGB x)const
	{
		RGB res = *this;
		return res -= x;
	}

	friend RGB operator-(float a, RGB x)
	{
		return RGB( a - x.r,a - x.g,a - x.b); 
	}

	RGB& operator*=(RGB x)
	{
		r *= x.r;
		g *= x.g;
		b *= x.b;
		return *this;
	}

	RGB operator*(RGB x)const
	{
		RGB res = *this;
		return res *= x;
	}

	RGB& operator*=(float a)
	{
		r *= a;
		g *= a;
		b *= a;
		return *this;
	}

	RGB operator*(float a)const
	{
		RGB res = *this;
		return res *= a;
	}

	friend RGB operator*(float a, RGB x)
	{
		return x * a;
	}

	RGB& operator/=(RGB x)
	{
		r /= x.r;
		g /= x.g;
		b /= x.b;
		return *this;
	}

	RGB operator/(RGB x)const
	{
		RGB res = *this;
		return res /= x;
	}

	RGB& operator/=(float a)
	{
		r /= a;
		g /= a;
		b /= a;
		return *this;
	}

	RGB operator/(float a)const
	{
		RGB res = *this;
		return res /= a;
	}

	RGB operator-() const
	{
		return { -r,-g,-b };
	}

	float average() const
	{
		return (r + g + b) / 3;
	}

	bool operator==(RGB x) const { return (r == x.r) && (g == x.g) && (b == x.b); }
	bool operator!=(RGB x) const { return (r != x.r) || (g != x.g) || (b != x.b); }

	float operator[](int i) const
	{
		if (i == 0)
		{
			return r;
		}
		else if (i == 1)
		{
			return g;
		}
		else if (i == 2)
		{
			return b;
		}
		else
		{
			return 0.0f;
		}
	}

	float& operator[](int i)
	{
		if (i == 0)
		{
			return r;
		}
		else if (i == 1)
		{
			return g;
		}
		else if (i == 2)
		{
			return b;
		}
		else
		{
			return 0.0f;
		}
	}
};


constexpr float Lambda_min = 360, Lambda_max = 830;
static constexpr int NSpectrumSamples = 4;
static constexpr float CIE_Y_INTEGRAL = 106.856895;

class SampledSpectrum
{
public:
	//构造函数支持两种 一个是为所有波长赋予同一个值 一个是把数组扔进去
	explicit SampledSpectrum(float v) { values.fill(v); }
	SampledSpectrum(const float* v)
	{
		for (int i = 0; i < NSpectrumSamples; i++)
		{
			values[i] = v[i];
		}
	}

	float operator[](int i)const { return values[i]; }
	float& operator[](int i) { return values[i]; }
	explicit operator bool()const {
		for (int i = 0; i < NSpectrumSamples; i++)
		{
			if (values[i] != 0) return true;
		}
		return false;
	}

	//向量化的四则运算，这对光谱来说很重要

	SampledSpectrum& operator+=(const SampledSpectrum& s)
	{
		for (int i = 0; i < NSpectrumSamples; i++)
		{
			values[i] += s.values[i];
		}
		return *this;
	}
	SampledSpectrum& operator-=(const SampledSpectrum& s)
	{
		for (int i = 0; i < NSpectrumSamples; i++)
		{
			values[i] -= s.values[i];
		}
		return *this;
	}
	SampledSpectrum& operator*=(const SampledSpectrum& s)
	{
		for (int i = 0; i < NSpectrumSamples; i++)
		{
			values[i] *= s.values[i];
		}
		return *this;
	}

	XYZ ToXYZ(const SampledWavelengths& lambda) const;
	RGB ToRGB(const SampledWavelengths& lambda, const RGBColorSpace& cs) const;

private:
	float values[NSpectrumSamples];
};

inline SampledSpectrum SafeDiv(SampledSpectrum s1, SampledSpectrum s2)
{
	SampledSpectrum r=SampledSpectrum(0);
	for (int i = 0; i < NSpectrumSamples; i++)
	{
		r[i] = (s2[i] != 0) ? s1[i] / s2[i] : 0.0f;
	}
	return r;
};

class SampledWavelengths
{
public:
	
	static SampledWavelengths SampleUinofrm(float u, float lambda_min = Lambda_min,float lambda_max = Lambda_max)
	{
		SampledWavelengths res;
		res.lambda[0] =lambda_min + u * (lambda_max-lambda_min);
		float delta= (lambda_max - lambda_min) / NSpectrumSamples;
		for (int i = 1; i < NSpectrumSamples; i++)
		{
			res.lambda[i] = res.lambda[i - 1] + delta;
			if (res.lambda[i] > lambda_max)
			{
				res.lambda[i] = lambda_min+res.lambda[i]- lambda_max;
			}
		}
		for (int i = 0; i < NSpectrumSamples; i++)
		{
			res.pdf[i] = 1.0f / (lambda_max - lambda_min);
		}
		return res;
	}

	float operator[](int i)const { return lambda[i]; }
	float &operator[](int i) { return lambda[i]; }
	SampledSpectrum PDF()const { return SampledSpectrum(pdf); }

	bool SecondaryTerminated() const
	{
		for (int i = 1; i < NSpectrumSamples; i++)
		{
			if (pdf[i] != 0)
			{
				return false;
			}
		}
		return true;
	}

	void TerminateSecondary()
	{
		if (SecondaryTerminated())
		{
			return;
		}
		for (int i = 1; i < NSpectrumSamples; i++)
		{
			pdf[i] = 0.0f;
		}
		pdf[0] /= NSpectrumSamples;
	}

private:
	float lambda[NSpectrumSamples];
	float pdf[NSpectrumSamples];
};



class RGBSigmoidPolynomial
{
public:
	float c0, c1, c2;

	RGBSigmoidPolynomial(float c0,float c1,float c2):c0(c0),c1(c1),c2(c2){}
	float operator()(float lambda) const
	{
		return s(c2+c1*lambda+c0*lambda*lambda);
	}


	float MaxValue() const
	{
		float res = std::max((*this)(360), (*this)(830));
		float lambda = -c1 / (2 * c0);
		if (lambda >= 360 && lambda <= 830)
		{
			res = std::max(res, (*this)(lambda));
		}
		return res;
	}
private:
	static float s(float x)
	{
		return 0.5f + x / (2 * std::sqrt(1 + x * x));
	}
};

class RGBSpectrumTable
{
public:
	static constexpr int res = 64;
	using CoefficientArray = float[3][res][res][res][3];

	const float* znodes;
	const CoefficientArray *coeffs;
	
	RGBSpectrumTable(const float* znodes, const CoefficientArray* coeffs):znodes(znodes),coeffs(coeffs){}

	RGBSigmoidPolynomial operator()(RGB rgb)const
	{
		if (rgb[0] == rgb[1] && rgb[1] == rgb[2])
		{
			return RGBSigmoidPolynomial(0, 0, (rgb[0] - 0.5f) / std::sqrt(rgb[0] * (1 - rgb[0])));
		}

		int maxc = (rgb[0] > rgb[1]) ? ((rgb[0] > rgb[2]) ? 0 : 2) : ((rgb[1] > rgb[2]) ? 1 : 2);
		float z = rgb[maxc];
		float x = rgb[(maxc + 1) % 3] * (res - 1) / z;
		float y = rgb[(maxc + 2) % 3] * (res - 1) / z;

		int xi = std::min((int)x, res - 2), yi = std::min((int)y, res - 2), zi = FindInterval(res, [&](int i) {return znodes[i] < z; });

		float dx = x - xi, dy = y - yi, dz = (z - znodes[zi]) / (znodes[zi + 1] - znodes[zi]);
		float c[3] = { 0.0f,0.0f,0.0f };

		int i = 0;
		for (i = 0; i < 3; i++)
		{
			auto co = [&](int dx, int dy, int dz)
			{
				return (*coeffs)[maxc][zi + dz][yi + dy][xi + dx][i];
			};
			c[i] = Interpolate(dz, Interpolate(dy, Interpolate(dx, co(0, 0, 0), co(1, 0, 0)), Interpolate(dx, co(0, 1, 0), co(1, 1, 0))),
				Interpolate(dy, Interpolate(dx, co(0, 0, 1), co(1, 0, 1)), Interpolate(dx, co(0, 1, 1), co(1, 1, 1))));
		}
		return RGBSigmoidPolynomial(c[0], c[1], c[2]);

	}
};

//每个波长都给出一个光谱值
class DenselySampledSpectrum
{
public:

	float values[Lambda_max - Lambda_min + 1];
	int lambda_min = Lambda_min;
	int lambda_max = Lambda_max;


	DenselySampledSpectrum(Spectrum s)
	{
		for (int i = lambda_min; i <= lambda_max; i++)
		{
			values[i - lambda_min] = s(i);
		}
	}

	//采样某个波长处的谱值
	float operator()(float lambda)const
	{
		int offset = std::lround(lambda) - lambda_min;
		if (offset < 0 || offset >= Lambda_max - Lambda_min + 1) return 0;
		return values[offset];
	}

	void Scale(float s)
	{
		for (int i = lambda_min; i <= lambda_max; i++)
		{
			values[i - lambda_min] *= s;
		}
	}

	SampledSpectrum Sample(const SampledWavelengths& lambda) const
	{
		SampledSpectrum s;
		for (int i = 0; i < NSpectrumSamples; i++)
		{
			int offset = std::lround(lambda[i]) - lambda_min;
			if (offset < 0 || offset >= Lambda_max - Lambda_min + 1)
			{
				s[i] = 0;
			}
			else
			{
				s[i] = values[offset];
			}
			
		}
		return s;
	}

	float MaxValue()const { return *std::max_element(values.begin(),values.end()); }

	bool operator==(const DenselySampledSpectrum& d) const
	{
		for (int i = lambda_min; i <= lambda_max; i++)
		{
			if (values[i - lambda_min] != d.values[i - lambda_min])
			{
				return false;
			}
		}
		return true;
	}

};


namespace Spectra {
	const DenselySampledSpectrum& X();
	const DenselySampledSpectrum& Y();
	const DenselySampledSpectrum& Z();
}

float InnerProduct(DenselySampledSpectrum a, Spectrum b)
{
	float sum = 0;
	for (int i = Lambda_min; i <= Lambda_max; i++)
	{
		sum += a(i)* b(i);
	}
	return sum;
}

XYZ SpectrumToXYZ(Spectrum s)
{
	return XYZ(InnerProduct(&Spectra::X(), s), InnerProduct(&Spectra::Y(), s), InnerProduct(&Spectra::Z(), s)) / CIE_Y_INTEGRAL;
};

XYZ SampledSpectrum::ToXYZ(const SampledWavelengths& lambda)const
{
	SampledSpectrum X = Spectra::X().Sample(lambda);
	SampledSpectrum Y = Spectra::Y().Sample(lambda);
	SampledSpectrum Z = Spectra::Z().Sample(lambda);

	SampledSpectrum pdf = lambda.PDF();
	return XYZ(SafeDiv(X * *this, pdf), SafeDiv(Y * *this, pdf), SafeDiv(Z * *this, pdf)) / CIE_Y_INTEGRAL;
};



class RGBColorSpace
{
public:

	vec2f r, g, b,w;
	Spectrum illuminant;
	const RGBSpectrumTable table;
	Mat3 XYZFromRGB, RGBFromXYZ;

	RGBColorSpace(vec2f r, vec2f g, vec2f b, Spectrum illuminant, const RGBSpectrumTable table):r(r),g(g),b(b),table(table),illuminant(illuminant)
	{
		XYZ W = SpectrumToXYZ(illuminant);
		w = W.xy();
		XYZ R = XYZ::FromxyY(r), G = XYZ::FromxyY(g), B = XYZ::FromxyY(b);


		Mat3 rgb(R.x, G.x, B.x, R.y, G.y, B.y, R.z, G.z, B.z);
		vec3f res= rgb.inverse() * vec3f(W.x, W.y, W.z);
		XYZ C = XYZ(res.x, res.y, res.z);

		XYZFromRGB = rgb * Mat3::Diag(C[0], C[1], C[2]);
		RGBFromXYZ = XYZFromRGB.inverse();
	}

	RGB ToRGB(XYZ xyz) const
	{
		vec3f q = vec3f(xyz.x, xyz.y, xyz.z);
		vec3f res = RGBFromXYZ * q;
		return RGB(res.x, res.y, res.z);
	}

	XYZ ToXYZ(RGB rgb) const
	{
		vec3f q = vec3f(rgb.r,rgb.g, rgb.b);
		vec3f res = XYZFromRGB * q;
		return XYZ(res.x, res.y, res.z);
	}



	RGBSigmoidPolynomial ToRGBCoeffs(RGB rgb) const;

};

RGBSigmoidPolynomial RGBColorSpace::ToRGBCoeffs(RGB rgb)const
{
	return table(rgb);
};

//这里的光谱指的是RGB光谱 我们不考虑其他的实现
class Spectrum
{
public:
	Spectrum(const RGBColorSpace& cs, RGB rgb)
	{
		p = cs.ToRGBCoeffs(rgb);
	}


	SampledSpectrum Sample(const SampledWavelengths& lambda) const {

		SampledSpectrum res;

		for (int i = 1; i < NSpectrumSamples; i++)
		{
			res[i] = p(lambda[i]);
		}
		return res;
	}

	float MaxValue()const
	{
		return p.MaxValue();
	}

	float operator()(float lambda) const
	{

		return p(lambda);
	}

private:
	RGBSigmoidPolynomial p;
};

RGB SampledSpectrum::ToRGB(const SampledWavelengths& lambda, const  RGBColorSpace& cs)const
{
	XYZ xyz = ToXYZ(lambda);
	return cs.ToRGB(xyz);
}