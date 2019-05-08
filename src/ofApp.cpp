#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	//shader.load(vert, frag);
	bool ok = shader.load("shaders/basic.vert.glsl", "shaders/basic.frag.glsl");
	printf("shader load ok %d\n", ok);

	

	float planeScale = 0.75;
	int planeWidth = ofGetWidth() * planeScale;
	int planeHeight = ofGetHeight() * planeScale;
	int planeGridSize = 20;
	int planeColumns = planeWidth / planeGridSize;
	int planeRows = planeHeight / planeGridSize;

	plane.set(planeWidth, planeHeight, planeColumns, planeRows, OF_PRIMITIVE_TRIANGLES); 
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){
	shader.begin();

	// center screen.
	float cx = ofGetWidth() / 2.0;
	float cy = ofGetHeight() / 2.0;

	// the plane is being position in the middle of the screen,
	// so we have to apply the same offset to the mouse coordinates before passing into the shader.
	float mx = mouseX - cx;
	float my = mouseY - cy;

	// we can pass in a single value into the shader by using the setUniform1 function.
	// if you want to pass in a float value, use setUniform1f.
	// if you want to pass in a integer value, use setUniform1i.
	shader.setUniform1f("mouseRange", 150); // SET A UNIFORM

	// we can pass in two values into the shader at the same time by using the setUniform2 function.
	// inside the shader these two values are set inside a vec2 object.
	shader.setUniform2f("mousePos", mx, my);  // SET A UNIFORM

	shader.setUniform1f("time", ofGetElapsedTimef());
	// color changes from magenta to blue when moving the mouse from left to right.
	float percentX = mouseX / (float)ofGetWidth();
	percentX = ofClamp(percentX, 0, 1);
	ofFloatColor colorLeft = ofColor::magenta;
	ofFloatColor colorRight = ofColor::blue;
	ofFloatColor colorMix = colorLeft.getLerped(colorRight, percentX);

	// create a float array with the color values.
	float mouseColor[4] = { colorMix.r, colorMix.g, colorMix.b, colorMix.a };

	// we can pass in four values into the shader at the same time as a float array.
	// we do this by passing a pointer reference to the first element in the array.
	// inside the shader these four values are set inside a vec4 object.
	shader.setUniform4fv("mouseColor", &mouseColor[0]);  // SET A UNIFORM


	// translate plane into center screen.
	float tx = ofGetWidth() / 2;
	float ty = ofGetHeight() / 2;
	ofTranslate(tx, ty);

	// the mouse/touch Y position changes the rotation of the plane.
	/*float percentY = mouseY / (float)ofGetHeight();
	float rotation = ofMap(percentY, 0, 1, -60, 60, true) + 60;
	ofRotate(rotation, 1, 0, 0);*/

	plane.drawWireframe();

	shader.end();

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
