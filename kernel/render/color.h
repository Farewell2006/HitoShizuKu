#pragma once

#include <math_utils.h>

class XYZ
{
public:
	XYZ(float x, float y,float z):x(x),y(y),z(z){}
	float average()const { return (x + y + z) / 3; }

	vec2f xy() const {
		return vec2f(x / (x + y + z), y / (x + y + z));

	}

	static XYZ FromxyY(vec2f xy, float y = 1.0f)
	{
		if (xy.y == 0)
		{
			return XYZ(0, 0, 0);
		}
		return XYZ(xy.y * Y / xy.y, Y, (1 - xy.x - xy.y) * Y / xy.y);
	}

	XYZ& operator+=(const XYZ &a)
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
	RGB(float r,float g,float b):r(r),g(g),b(b){}

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
		return { a - x.r,a - x.g,a - x.b; }
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
		g *=a;
		b *= a;
		return *this;
	}

	RGB operator*(float a)const
	{
		RGB res = *this;
		return res *=a;
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
		return res/= a;
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

	float &operator[](int i)
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

