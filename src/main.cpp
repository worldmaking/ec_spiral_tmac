#include "ofMain.h"
#include "ofApp.h"


//========================================================================
int main( ){
	//ofSetupOpenGL(1024,768,OF_WINDOW);			// <-------- setup the GL context
	ofGLWindowSettings settings;
	settings.setGLVersion(3, 2); /// < select your GL Version here
	settings.setSize(900, 900);
	ofCreateWindow(settings); ///< create your window here

	ofRunApp(new ofApp());
}
