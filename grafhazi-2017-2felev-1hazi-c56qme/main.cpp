//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Magyar Balazs
// Neptun : C56QME
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

//--------------------------------------------------------
// 3D Vektor - forras a tavalyi sablon
//--------------------------------------------------------
struct Vector {
	float x, y, z;

	Vector() {
		x = y = z = 0;
	}
	Vector(float x0, float y0, float z0 = 0) {
		x = x0; y = y0; z = z0;
	}
	Vector operator*(float a) {
		return Vector(x * a, y * a, z * a);
	}
	Vector operator+(const Vector& v) {
		return Vector(x + v.x, y + v.y, z + v.z);
	}
	Vector operator-(const Vector& v) {
		return Vector(x - v.x, y - v.y, z - v.z);
	}
	float operator*(const Vector& v) { 	// dot product
		return (x * v.x + y * v.y + z * v.z);
	}
	Vector operator%(const Vector& v) { 	// cross product
		return Vector(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}
	Vector operator/(float a) {
		return Vector(x / a, y / a, z / a);
	}
	float Length() { return sqrt(x * x + y * y + z * z); }
};


float spacepressedtime = 0;
//--------------------------------------------------------
// Spektrum illetve szin -szinten a tavalyi sablonbol
//--------------------------------------------------------
struct Color {
	float r, g, b;

	Color() {
		r = g = b = 0;
	}
	Color(float r0, float g0, float b0) {
		r = r0; g = g0; b = b0;
	}
	Color operator*(float a) {
		return Color(r * a, g * a, b * a);
	}
	Color operator*(const Color& c) {
		return Color(r * c.r, g * c.g, b * c.b);
	}
	Color operator+(const Color& c) {
		return Color(r + c.r, g + c.g, b + c.b);
	}
};

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 130
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	in vec3 vertexColor;	    // variable input from Attrib Array selected by glBindAttribLocation
	out vec3 color;				// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 130
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {

public:
	float m[4][4];
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
};

float getDistance(Vector p1, Vector p2)
{
	float distance = pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2);
	distance = sqrt(distance);
	return distance;
}

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class BezierCurve {
public:
	float getHeight(float x, float y) { //0 és 1 közt legyen a floatok mérete

		mat4 gbx(
			0.5, 0.5, 0.5, 0.5,
			0, 1, 1, 0,
			0, 1, 1, 0.5,
			0.5, 0, 0, 0); // GbX matrix

		float ret = 0;

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				ret += gbx.m[i][j] * getB(x, i)*getB(y, j);
			}
		}
		return ret*1.8;
	}
	float getB(float u, int no) //u számra a no-adik b-t adja vissza
	{
		switch (no)
		{
		case 0:
			return pow((1 - u), 3);
			break;
		case 1:
			return 3 * u*pow((1 - u), 2);
			break;
		case 2:
			return 3 * pow(u, 3)*(1 - u);
			break;
		case 3:
			return pow(u, 3);
		default:
			break;
		}
	}
};


class Triangle {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
public:
	Triangle() {
		Animate(0);
	}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { -8, -8, -6, 10, 8, -2 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords),  // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 1;// *sinf(t);
		sy = 1;// *cosf(t);
		wTx = 0;// 4 * cosf(t / 2);
		wTy = 0;// 4 * sinf(t / 2);
	}

	void Draw() {
		mat4 M(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = M * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};

class Map {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
	float fir;
	int Ag;
	int mapCoordsLoc;
	int mapColorsLoc;
public:
	Map() {
		Animate(0);
	}

	void Create() {
		BezierCurve b;
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects
									//coords
		float mapCoords[20000];
		//color
		float mapColors[40000];
		mapCoordsLoc = 0;
		mapColorsLoc = 0;

		float coordOnMapX = -5;
		float coordOnMapY = -5;
		float yellowIntensity = 0.4;

		for (coordOnMapX = -10; coordOnMapX < 10; coordOnMapX += 0.5)
		{

			for (coordOnMapY = -10; coordOnMapY < 10; coordOnMapY += 0.5)
			{
				//elso háromszög
				mapCoords[mapCoordsLoc] = coordOnMapX;
				mapCoordsLoc++;
				mapCoords[mapCoordsLoc] = coordOnMapY;
				mapCoordsLoc++;
				//színezés
				mapColors[mapColorsLoc] = b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20);
				mapColorsLoc++;
				mapColors[mapColorsLoc] = 1 - (b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20));
				mapColorsLoc++;
				mapColors[mapColorsLoc] = yellowIntensity;
				mapColorsLoc++;

				coordOnMapY += 0.5;

				mapCoords[mapCoordsLoc] = coordOnMapX;
				mapCoordsLoc++;
				mapCoords[mapCoordsLoc] = coordOnMapY;
				mapCoordsLoc++;
				//színezés
				mapColors[mapColorsLoc] = b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20);
				mapColorsLoc++;
				mapColors[mapColorsLoc] = 1 - (b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20));
				mapColorsLoc++;
				mapColors[mapColorsLoc] = yellowIntensity;
				mapColorsLoc++;

				coordOnMapY -= 0.5;
				coordOnMapX += 0.5;

				mapCoords[mapCoordsLoc] = coordOnMapX;
				mapCoordsLoc++;
				mapCoords[mapCoordsLoc] = coordOnMapY;
				mapCoordsLoc++;
				//színezés
				mapColors[mapColorsLoc] = b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20);
				mapColorsLoc++;
				mapColors[mapColorsLoc] = 1 - (b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20));
				mapColorsLoc++;
				mapColors[mapColorsLoc] = yellowIntensity;
				mapColorsLoc++;

				//coordOnMapX -= 0.5;


				//masodik háromszög
				mapCoords[mapCoordsLoc] = coordOnMapX;
				mapCoordsLoc++;
				mapCoords[mapCoordsLoc] = coordOnMapY;
				mapCoordsLoc++;
				//színezés
				mapColors[mapColorsLoc] = b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20);
				mapColorsLoc++;
				mapColors[mapColorsLoc] = 1 - (b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20));
				mapColorsLoc++;
				mapColors[mapColorsLoc] = yellowIntensity;
				mapColorsLoc++;

				coordOnMapY += 0.5;

				mapCoords[mapCoordsLoc] = coordOnMapX;
				mapCoordsLoc++;
				mapCoords[mapCoordsLoc] = coordOnMapY;
				mapCoordsLoc++;
				//színezés
				mapColors[mapColorsLoc] = b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20);
				mapColorsLoc++;
				mapColors[mapColorsLoc] = 1 - (b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20));
				mapColorsLoc++;
				mapColors[mapColorsLoc] = yellowIntensity;
				mapColorsLoc++;

				coordOnMapX -= 0.5;

				mapCoords[mapCoordsLoc] = coordOnMapX;
				mapCoordsLoc++;
				mapCoords[mapCoordsLoc] = coordOnMapY;
				mapCoordsLoc++;
				//színezés
				mapColors[mapColorsLoc] = b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20);
				mapColorsLoc++;
				mapColors[mapColorsLoc] = 1 - (b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20));
				mapColorsLoc++;
				mapColors[mapColorsLoc] = yellowIntensity;
				mapColorsLoc++;



				coordOnMapY -= 0.5;
				//std::cout << (b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20))<<"\n";

			}
		}

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(float)*(mapCoordsLoc),  // number of the vbo in bytes
			mapCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*(mapColorsLoc), mapColors, GL_STATIC_DRAW);	// copy to the GPU

																									// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 1;// *sinf(t);
		sy = 1;// *cosf(t);
		wTx = 0;// 4 * cosf(t / 2);
		wTy = 0;// 4 * sinf(t / 2);
	}

	void Draw() {
		mat4 scale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // scale and position matrix
		mat4 rot(cos(fir), -sin(fir), 0, 0,
			sin(fir), cos(fir), 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // rotate matrix
		mat4 M = rot * scale;
		mat4 MVPTransform = M * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, mapColorsLoc / 3);	// draw a single triangle with vertices defined in vao
	}
};

class LagrangeCurve {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[10000]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
	std::vector<Vector> cps; //control points
	std::vector<float> ts; //knot values

public:
	std::vector<float> times;
	LagrangeCurve() {
		nVertices = 0;
	}

	float L(int i, float t) {
		float Li = 1.0f;
		for (int j = 0; j < cps.size(); j++)
			if (j != i)
				Li = Li * (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}

	void AddControlPoint(float cpx, float cpy, float time) {
		vec4 wVertex = vec4(cpx, cpy, 0, 1) * camera.Pinv() * camera.Vinv();
		Vector v;
		v.x = wVertex.v[0];
		v.y = wVertex.v[1];
		v.z = 0.0f;
		float ti = cps.size(); 	// or something better
		cps.push_back(v); ts.push_back(ti);
		times.push_back(time);
	}

	Vector r(float t) { //in io 0es1 kozt
		Vector rr(0, 0, 0);
		for (int i = 0; i < cps.size(); i++)
			rr = rr + (cps[i] * L(i, t));
		return rr;
	}


	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(float cX, float cY) {
		if (nVertices >= 20) return;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices] = wVertex.v[0];
		vertexData[5 * nVertices + 1] = wVertex.v[1];
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 0; // blue
		nVertices++;



		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Recalculate()
	{

		BezierCurve b;
		float res = 0.01; //resolution
		nVertices = 0;
		float distance = 0;
		Vector elozo;
		for (float f = 0; f < cps.size() - 1; f += res)
		{
			Vector jelenlegi;
			jelenlegi.x = r(f).x;
			jelenlegi.y = r(f).y;
			jelenlegi.z = b.getHeight((jelenlegi.x + 10) / 20, (jelenlegi.y + 10) / 20) * 20;
			//std::cout << jelenlegi.z << "\n";
			if (f != 0)
			{
				distance += getDistance(jelenlegi, elozo);
			}
			elozo = jelenlegi;

			vertexData[5 * nVertices] = jelenlegi.x;
			vertexData[5 * nVertices + 1] = jelenlegi.y;
			vertexData[5 * nVertices + 2] = 1; // red
			vertexData[5 * nVertices + 3] = 1; // green
			vertexData[5 * nVertices + 4] = 1; // blue
			nVertices++;


		}
		std::cout << distance * 50 << "\n";
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}
};
LagrangeCurve lagrange;

class Biker {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
	float fir;
	int Ag;
	int mapCoordsLoc;
	int mapColorsLoc;
public:
	Biker() {
		Animate(0);
	}

	void Create() {
		BezierCurve b;
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects
									//coords
		float mapCoords[20000];
		//color
		float mapColors[40000];
		mapCoordsLoc = 0;
		mapColorsLoc = 0;

		float coordOnMapX = 0;
		float coordOnMapY = 0;

		//elso háromszög
		mapCoords[mapCoordsLoc] = coordOnMapX;
		mapCoordsLoc++;
		mapCoords[mapCoordsLoc] = coordOnMapY;
		mapCoordsLoc++;
		//színezés
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 1;
		mapColorsLoc++;

		coordOnMapY += 0.5;

		mapCoords[mapCoordsLoc] = coordOnMapX;
		mapCoordsLoc++;
		mapCoords[mapCoordsLoc] = coordOnMapY;
		mapCoordsLoc++;
		//színezés
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 1;
		mapColorsLoc++;

		coordOnMapY -= 0.7;
		coordOnMapX -= 0.2;

		mapCoords[mapCoordsLoc] = coordOnMapX;
		mapCoordsLoc++;
		mapCoords[mapCoordsLoc] = coordOnMapY;
		mapCoordsLoc++;
		//színezés
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 1;
		mapColorsLoc++;

		coordOnMapX += 0.4;


		//masodik háromszög
		mapCoords[mapCoordsLoc] = coordOnMapX;
		mapCoordsLoc++;
		mapCoords[mapCoordsLoc] = coordOnMapY;
		mapCoordsLoc++;
		//színezés
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 1;
		mapColorsLoc++;

		coordOnMapY += 0.7;
		coordOnMapX -= 0.2;

		mapCoords[mapCoordsLoc] = coordOnMapX;
		mapCoordsLoc++;
		mapCoords[mapCoordsLoc] = coordOnMapY;
		mapCoordsLoc++;
		//színezés
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 1;
		mapColorsLoc++;

		coordOnMapY -= 0.5;

		mapCoords[mapCoordsLoc] = coordOnMapX;
		mapCoordsLoc++;
		mapCoords[mapCoordsLoc] = coordOnMapY;
		mapCoordsLoc++;
		//színezés
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 0;
		mapColorsLoc++;
		mapColors[mapColorsLoc] = 1;
		mapColorsLoc++;

		//std::cout << (b.getHeight((coordOnMapX + 10) / 20, (coordOnMapY + 10) / 20))<<"\n";

// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(float)*(mapCoordsLoc),  // number of the vbo in bytes
			mapCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*(mapColorsLoc), mapColors, GL_STATIC_DRAW);	// copy to the GPU

																								// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) { //az eltelt ido a space lenyomasa ota
		sx = 1; // *sinf(t);
		sy = 1; // *cosf(t);
		Vector loc;
		if (spacepressedtime != 0) {
			//ha vége a pályának, marad ott
			loc = lagrange.r(lagrange.times.size() - 1);
			for (int i = 0; i < lagrange.times.size() - 1; i++)
			{
				if ((lagrange.times[i] - lagrange.times[0]) < t && t < (lagrange.times[i + 1] - lagrange.times[0])) // ha a megfelelo szakaszban van
				{
					float elsokatt = lagrange.times[i] - lagrange.times[0];
					float masodikkatt = lagrange.times[i + 1] - lagrange.times[0];
					float elteltido = masodikkatt - elsokatt;
					float state = i + (t-elsokatt) / elteltido;
					loc = lagrange.r(state);
				}
			}
			wTx = loc.x;// 4 * cosf(t / 2);
			wTy = loc.y;// 4 * sinf(t / 2);
		}
		else
		{
			wTx = 0;// 4 * cosf(t / 2);
			wTy = 0;// 4 * sinf(t / 2);
		}
	}

	void Draw() {
		mat4 scale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // scale and position matrix
		mat4 rot(cos(fir), -sin(fir), 0, 0,
			sin(fir), cos(fir), 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // rotate matrix
		mat4 M = rot * scale;
		mat4 MVPTransform = M * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, (mapCoordsLoc / 2) - 0);	// draw a single triangle with vertices defined in vao
	}
};

class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[100]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:
	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(float cX, float cY) {
		if (nVertices >= 20) return;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices] = wVertex.v[0];
		vertexData[5 * nVertices + 1] = wVertex.v[1];
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 0; // blue
		nVertices++;
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}
};




// The virtual world: collection of two objects
//Triangle triangle;
//LineStrip lineStrip;
Map mapTriangles;

Biker PepsiBela;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
//	triangle.Create();
	lagrange.Create();
	PepsiBela.Create();
	static float color[3] = { 0.95, 1.0, 1.0 };
	mapTriangles.Create();

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect Attrib Arrays to input variables of the vertex shader
	glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
	glBindAttribLocation(shaderProgram, 1, "vertexColor");    // vertexColor gets values from Attrib Array 1

															  // Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

//	triangle.Draw();
	mapTriangles.Draw();
	//lineStrip.Draw();
	lagrange.Draw();
	PepsiBela.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == ' ') spacepressedtime = glutGet(GLUT_ELAPSED_TIME);
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		//lineStrip.AddPoint(cX, cY);
		lagrange.AddControlPoint(cX, cY, glutGet(GLUT_ELAPSED_TIME));
		lagrange.Recalculate();
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the mapt of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera
//	triangle.Animate(sec);					// animate the triangle object
	PepsiBela.Animate(glutGet(GLUT_ELAPSED_TIME) - spacepressedtime);
	glutPostRedisplay();					// redraw the scene
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
