#include "ofMain.h"
#include "ofApp.h"


//========================================================================
int main( ){
	//ofSetupOpenGL(1024,768,OF_WINDOW);			// <-------- setup the GL context
	ofGLWindowSettings settings;
	settings.setGLVersion(4, 1); /// < select your GL Version here

	auto win = ofCreateWindow(settings); ///< create your window here
	win->setVerticalSync(false);
	ofSetFrameRate(90);
	ofRunApp(new ofApp());
}
