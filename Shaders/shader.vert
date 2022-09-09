#version 330

layout(location = 0) in uint state;
varying vec4 vColor;

uniform vec4 _uOnColor;
uniform vec4 _uOffColor;
uniform uint _uWidthX;
uniform uint _uWidthY;

uniform vec4 _uWindowXY;

uint xId;
uint yId;
float xmin; float xmax;
float ymin; float ymax;
float dx; float dy;

uint store;
void main(void) 
{
	store = uint(floor(gl_VertexID/_uWidthX));
	xId = gl_VertexID - (_uWidthX * store);
	yId = store;

	dx = (_uWindowXY.y - _uWindowXY.x) / float(_uWidthX);
	dy = (_uWindowXY.w - _uWindowXY.z) / float(_uWidthY);

	gl_Position = vec4(float(_uWindowXY.x)+dx/2 + xId * dx, float(_uWindowXY.z)+dy/2 + yId * dy, 0., 1.0);
	vColor = vec4(
				_uOnColor.x * float(state) + _uOffColor.x * (1 - float(state)),
				_uOnColor.y * float(state) + _uOffColor.y * (1 - float(state)),
				_uOnColor.z * float(state) + _uOffColor.z * (1 - float(state)), 
				1.);

};