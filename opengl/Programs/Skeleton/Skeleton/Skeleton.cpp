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
// Nev    : Burján Viktor
// Neptun : MLTYU6
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

#if defined(__APPLE__)
#include <OpenGL/glew.h>
#include <OpenGL/freeglut.h>
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
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m02; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m02; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m02; m[3][3] = m33;
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

	vec4 operator*(const float f) {
		vec4 result;
		for (int i = 0; i < 4; i++) {
			result.v[i] = v[i] * f;
		}
		return result;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	float operator*(const vec4& mat) {
		float result;
			result = v[0] * mat.v[0] + v[1] * mat.v[1] + v[2] * mat.v[2];
		return result;
	}

	vec4 operator+(const vec4& mat) {
		float x, y, z, w;
		x = v[0] + mat.v[0]; 
		y = v[1] + mat.v[1]; 
		z = v[2] + mat.v[2]; 
		w = v[3] + mat.v[3];
		return vec4(x, y, 0, 1);
	}


	vec4 operator-(const vec4& mat) {
		float x, y, z, w;
		x = v[0] - mat.v[0]; 
		y = v[1] - mat.v[1]; 
		z = v[2] - mat.v[2];
		w = 1;
		return vec4(x, y, z, 1);
	}
	
	vec4 operator/(const float oszto) {
		return vec4(v[0] / oszto, v[1] / oszto, 0, 1);
	}
};

float len(vec4 a, vec4 b) {
	float hossz = sqrtf((b.v[0] - a.v[0]) * (b.v[0] - a.v[0]) - (b.v[1] - a.v[1]) * (b.v[1] - a.v[1]));
	return hossz;
}

float len(vec4 a) {
	float hossz = sqrtf(a.v[0] * a.v[0] + a.v[1] * a.v[1]);
	return hossz;
}


// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	void setCenter(float x, float y) {
		wCx = x;
		wCy = y;
	}
	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1,    0, 0, 0,
			        0,    1, 0, 0,
			        0,    0, 1, 0,
			     -wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2/wWx,    0, 0, 0,
			        0,    2/wWy, 0, 0,
			        0,        0, 1, 0,
			        0,        0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1,     0, 0, 0,
				    0,     1, 0, 0,
			        0,     0, 1, 0,
			        wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx/2, 0,    0, 0,
			           0, wWy/2, 0, 0,
			           0,  0,    1, 0,
			           0,  0,    0, 1);
	}

	void Animate(float t) {
		wCx = 0; //10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;


struct Csomopont {
	vec4 koord;
	vec4 seb;
	float t;
};

Csomopont csomopontok[11];



vec4 a0(int i) {
	return vec4(csomopontok[i].koord.v[0], csomopontok[i].koord.v[1], 0, 1);
}

vec4 a1(int i) {
	return vec4(csomopontok[i].seb.v[0], csomopontok[i].seb.v[1], 0, 1);
}

vec4 a2(int i) {
	float nevezo = csomopontok[i + 1].t - csomopontok[i].t;
	vec4 ujVektor1 = ((csomopontok[i + 1].koord - csomopontok[i].koord) * 3) / (nevezo * nevezo);
	vec4 ujVektor2 = (csomopontok[i + 1].seb + (csomopontok[i].seb * 2)) / nevezo;
	return ujVektor1 - ujVektor2;
}

vec4 a3(int i) {
	float nevezo = csomopontok[i + 1].t - csomopontok[i].t;
	vec4 ujVektor1 = ((csomopontok[i].koord - csomopontok[i + 1].koord) * 2) / (nevezo * nevezo * nevezo);
	vec4 ujVektor2 = ((csomopontok[i + 1].seb + csomopontok[i].seb)) / (nevezo * nevezo);
	return ujVektor1 + ujVektor2;
}

vec4 r_t(int i, float t) {
	float nevezo = t - csomopontok[i].t;
	vec4 asd = ((a3(i) * nevezo * nevezo * nevezo) +
		(a2(i) * nevezo * nevezo) +
		(a1(i) * nevezo)) + a0(i);
	return asd;
}

int nVertices = 0;

vec4 sebesseg(int i) {
	

	if (i == 0) {
		float nevezo1 = csomopontok[1].t - csomopontok[0].t;
		float nevezo2 = csomopontok[0].t - csomopontok[nVertices + 1].t;
		//printf("sebesseg fuggvenyben: %f  ---  %f ", csomopontok[1].koord.v[1], csomopontok[0].koord.v[1]);
		return ((csomopontok[1].koord - csomopontok[i].koord) / nevezo1 +
			(csomopontok[0].koord - csomopontok[nVertices + 1].koord) / nevezo2) * 0.9;
	}
	if (i == nVertices + 1) {
		float nevezo1 = csomopontok[0].t - csomopontok[i].t;
		float nevezo2 = csomopontok[i].t - csomopontok[i - 1].t;
		//printf("sebesseg fuggvenyben: %f  ---  %f ", csomopontok[0].koord.v[1], csomopontok[i].koord.v[1]);
		return ((csomopontok[0].koord - csomopontok[i].koord) / nevezo1 +
			(csomopontok[i].koord - csomopontok[i - 1].koord) / nevezo2) * 0.9;
	}


	else {
		float nevezo1 = csomopontok[i + 1].t - csomopontok[i].t;
		float nevezo2 = csomopontok[i].t - csomopontok[i - 1].t;
		//printf("sebesseg fuggvenyben: %f  ---  %f ", csomopontok[i + 1].koord.v[1], csomopontok[i].koord.v[1]);
		return ((csomopontok[i + 1].koord - csomopontok[i].koord) / nevezo1 +
			(csomopontok[i].koord - csomopontok[i - 1].koord) / nevezo2) * 0.9;
	}
}
float tomeg1 = 3;
float tomeg2 = 2;

float allando = 0.006f;

class Star {
protected:
	float tomeg = 10;
public:
	unsigned int vao; //Vertex array object id
	float sx, sy; //SCALING-re van
	float wTx, wTy; 
	unsigned int vbo[2];

	float r = 2.0;

	float rotX = 1;
	float rotY = 1;
	float red = 1;
	float green = 1;
	float blue = 0;

	float center = 0;
	Star() {
		Animate(0);
	}

	float getX() {
		return wTx;
	}
	float getY() {
		return wTy;
	}

	void setCenter(float newCenter) {
		center = newCenter;
	}

	void setR(float R) {
		r = R;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);


		glGenBuffers(2, &vbo[0]);

		//vertex koordináták:
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		static float vertexCoords[] = {
			center - (r / 2), center - (r / 2),                         //14
			center - (r / 2) , center + (r / 2),                  //1
			center + (r / 2) , center + (r / 2),                  //5
			center + (r / 2), center - (r / 2),                       //9
			center - (r / 2), center - (r / 2),                         //14
			center - (r / 6) , center + (r / 2),                    //2
			center + (r / 6) , center + (r / 2),                    //4
			center             , center + (r - r * (1.0 / 3.0)) ,   //3
			center + (r - r * (1.0 / 3.0)), center,                 //7
			center           , center - (r - r * (1.0 / 3.0)),      //11
			center - (r - r * (1.0 / 3.0)), center,                 //16
			center             , center + (r - r * (1.0 / 3.0))             //3
		};
		glBufferData(
			GL_ARRAY_BUFFER,
			sizeof(vertexCoords),
			vertexCoords,
			GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(
			0,
			2, GL_FLOAT,
			GL_FALSE,
			0, NULL);
		//Eddig voltak a koordináták, innen jönnek a színek

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		static float vertexColors[] = {
			red, green, blue,  red, green, blue,   red, green, blue,   red, green, blue,   red, green, blue,
			red, green, blue,  red, green, blue,   red, green, blue,   red, green, blue,   red, green, blue,
			red, green, blue,  red, green, blue,   red, green, blue
		};
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	virtual void Animate(float time) {
		float t = fmod(time, csomopontok[nVertices].t - csomopontok[0].t) + csomopontok[0].t;
		int i = 0;
		if (nVertices > 1) {
			for (; i < nVertices+1; i++)
				if (csomopontok[i].t <= t && t <= csomopontok[i + 1].t)
					break;

				vec4 e = r_t(i, t);
				//printf("koordinatak: : : %f , %f \n", e.v[0], e.v[1]);
				wTx = e.v[0]; 
				wTy = e.v[1];
			
				rotX = cos(time / 800);
				rotY = sinf(time / 800);
		}
		sx = 0.6 + fabs(sinf(time/220.0)); // *sinf(t);
		sy = 0.6 + fabs(sinf(time/220.0)); // *cosf(t);
		//wTx = 0; // 4 * cosf(t / 2);
		//wTy = 0; // 4 * sinf(t / 2);
	}
	void Draw() {

		mat4 Scale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1);

		mat4 Trans(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wTx, wTy, 0, 1);


		mat4 Rotate(rotX, rotY, 0, 0,
			-rotY, rotX, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		mat4 MVPTransform = Scale * Rotate * Trans * camera.V() * camera.P();
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao); // make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 12); // draw a single triangle with vertices defined in vao

	}
};

class Star2 {
protected:
	float tomeg = 2;
public:
	unsigned int vao; //Vertex array object id
	float sx, sy; //SCALING-re van
	float wTx, wTy; // translation
	unsigned int vbo[2];

	float rotX = 0;
	float rotY = 0;

	vec4 v = vec4(0.0,0.0,0.0,1.0);

	float r = 1.0;

	float red = 0;
	float green = 1;
	float blue = 1;

	float center = 0;
	Star2() {
		Animate(0);
	}

	float getX() {
		return wTx;
	}
	float getY() {
		return wTy;
	}

	void setCenter(float newCenter) {
		center = newCenter;
	}

	void setR(float R) {
		r = R;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);


		glGenBuffers(2, &vbo[0]);

		//vertex koordináták:
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		static float vertexCoords[] = {
			center - (r / 2), center - (r / 2),                         //14
			center - (r / 2) , center + (r / 2),                  //1
			center + (r / 2) , center + (r / 2),                  //5
			center + (r / 2), center - (r / 2),                       //9
			center - (r / 2), center - (r / 2),                         //14
			center - (r / 6) , center + (r / 2),                    //2
			center + (r / 6) , center + (r / 2),                    //4
			center             , center + (r - r * (1.0 / 3.0)) ,   //3
			center + (r - r * (1.0 / 3.0)), center,                 //7
			center           , center - (r - r * (1.0 / 3.0)),      //11
			center - (r - r * (1.0 / 3.0)), center,                 //16
			center             , center + (r - r * (1.0 / 3.0))             //3
		};
		glBufferData(
			GL_ARRAY_BUFFER,
			sizeof(vertexCoords),
			vertexCoords,
			GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(
			0,
			2, GL_FLOAT,
			GL_FALSE,
			0, NULL);
		//Eddig voltak a koordináták, innen jönnek a színek

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		static float vertexColors[] = {
			red, green, blue,  red, green, blue,   red, green, blue,   red, green, blue,   red, green, blue,
			red, green, blue,  red, green, blue,   red, green, blue,   red, green, blue,   red, green, blue,
			red, green, blue,  red, green, blue,   red, green, blue
		};
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void setV(vec4 nagyCsillag) {
		vec4 e = nagyCsillag - vec4(wTx, wTy, 0, 1);
		float len = sqrtf(e.v[0] * e.v[0] + e.v[1] * e.v[1]);
		if (len > 0.05) {
			vec4 k = e * ((tomeg1 * tomeg2) / (len *len *len)) * allando; //printf("mozgas vektora: %f , %f \n", e.v[0], e.v[1]);
			v = k / tomeg2 - (v*0.08);
		}
		else {
			v = v*(-1);
		}
	}
	void Animate(float time) {
		wTx += v.v[0];
		wTy += v.v[1];
		sx = 0.3 + fabs(sinf(time / 180.0)); 
		sy = 0.3 + fabs(sinf(time / 180.0));

		rotX = cos(time / 800);
		rotY = sinf(time / 800);
	}

	void Draw() {
		mat4 Scale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1);

		mat4 Trans(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wTx, wTy, 0, 1);


		mat4 Rotate(rotX, rotY, 0, 0,
			-rotY, rotX, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		mat4 MVPTransform = Scale * Rotate * Trans * camera.V() * camera.P();
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao); // make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 12); // draw a single triangle with vertices defined in vao

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

		// vertex coordinates
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { -8, -8, -6, 10, 8, -2 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			         sizeof(vertexCoords), // number of the vbo in bytes
					 vertexCoords,		   // address of the data array on the CPU
					 GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0); 
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			                  2, GL_FLOAT,  // components/attribute, component type
							  GL_FALSE,		// not in fixed point format, do not normalized
							  0, NULL);     // stride and offset: it is tightly packed

		// vertex colors
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

		// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
		// Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 0.5f /**sinf(t)*/;
		sy = 0.5f /**cosf(t)*/;
		wTx = 0; // 4 * cosf(t / 2);
		wTy = 0; // 4 * sinf(t / 2);
	}

	void Draw() {
		mat4 M(sx,   0,  0, 0,
			    0,  sy,  0, 0,
			    0,   0,  0, 0,
			  wTx, wTy,  0, 1); // model matrix

		mat4 MVPTransform = M * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};



class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[4400]; // interleaved data of coordinates and colors
	float vertexData2[1000];
	int k = 5;
public:

	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		GLuint vbo;	// vertex/index buffer object
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

	void AddPoint(float cX, float cY, float time) {
		if (nVertices >= 10) return;
		//printf("1 ---- A ket koordinata az egertol: %f es %f \n", cX, cY);

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();

		csomopontok[nVertices].koord = wVertex;
		csomopontok[nVertices].t = time;// printf("%f \n", time);
		if (nVertices > 0) {
			for (int i = 0; i <= nVertices; i++) {
				csomopontok[i].seb = sebesseg(i);
			}
		}

		csomopontok[nVertices + 1].koord = csomopontok[0].koord;
		csomopontok[nVertices + 1].seb = csomopontok[0].seb;
		csomopontok[nVertices + 1].t = csomopontok[nVertices].t + 500;

		for (int i = 0; i < nVertices + 1; i++) {
			int a = 0;
			for (float t = csomopontok[i].t; t <= csomopontok[i + 1].t; t += (csomopontok[i + 1].t - csomopontok[i].t) / 80 ) {
				vec4 e = r_t(i, t);
				vertexData[400 * i + a * 5] = e.v[0];
				vertexData[400 * i + a * 5 + 1] = e.v[1];
				vertexData[400 * i + a * 5 + 2] = 1;
				vertexData[400 * i + a * 5 + 3] = 1;
				vertexData[400 * i + a * 5 + 4] = 1;
				a++;
			}
		}


		nVertices++;
		glBufferData(GL_ARRAY_BUFFER, (nVertices) * 400 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw() {
		if (true) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, (nVertices)*80);
		}
	}
};


// The virtual world: collection of two objects
Triangle triangle;
LineStrip lineStrip;
Star star;
Star2 star2;
Star2 star3;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	star.Create();
	star2.Create();
	star3.Create();

	star3.wTx = 7;
	star3.wTy = 7;

	//triangle.Create();
	lineStrip.Create();

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

	star.Draw();

	//triangle.Draw();
	
	lineStrip.Draw();

	star2.Draw();
	star3.Draw();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == ' ') {
		//printf("\n asdasdasd ");
		camera.wCx = star.wTx;
		camera.wCy = star.wTy;
		//camera.setCenter(star.getX(), star.getY());
		//glutPostRedisplay();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		camera.setCenter(0, 0);
	}
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		lineStrip.AddPoint(cX, cY, glutGet(GLUT_ELAPSED_TIME));
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	//camera.Animate(sec);					// animate the camera
	triangle.Animate(sec);					// animate the triangle object
	glutPostRedisplay();					// redraw the scene

	star.Animate(time);
	
	star2.setV(vec4(star.wTx, star.wTy, 0, 1));
	star3.setV(vec4(star.wTx, star.wTy, 0, 1));

	//star2.setV(vec4(star.getX(), star.getY(), 0, 1));
	//star3.setV(vec4(star.getX(), star.getY(), 0, 1));
	star2.Animate(time);
	star3.Animate(time); //printf(" --- %f, %f \n ", star3.wTx, star3.wTy);
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);

	glutInitContextVersion(majorVersion, minorVersion);
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);  // 8 bit R,G,B,A + double buffer + depth buffer
	glutCreateWindow(argv[0]);
	glewExperimental = true;
	glewInit();

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
