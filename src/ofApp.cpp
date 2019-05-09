#include "ofApp.h"

#include "al_glm.h"
#include "al_math.h"
#include "an_field3d.h"
#include "an_math.h"
#include "genlib.h"
#include <algorithm>

#define STRINGIFY(A) #A
//--------------------------------------------------------------
void ofApp::setup(){

	world.setup();
	mLandtexmix = 0.f;
	mLandtexmixTarget = 0.f;

	mSleepMinMS = 0.;	// background threads must sleep at least this no. of ms in order to let main thread have space
	mCanceled = false;
	mScreenshot = false;

	mNavKeyRate = glm::vec3(10.f);
	mNavKeyTurnRate = glm::vec3(1.f);
	mNavSpeed = 0.2f;

	mShowParticles = true;
	mShowWorld = true;
	mShowOrganisms = true;
	mShowCreatures = true;
	mShowControllers = true;
	mShowStalks = true;

	mOrganismInstanceCount = 0;
	mStalkInstanceCount = 0;

	mAutoScreenshotTime = 0;
	mEvaluationNoticeTime = 0;

	mFaceOrient = mNavOrient = glm::quat_cast(glm::mat4(1.0));

#ifdef AN_FINAL
	mStartInFullscreen = true;
	mPerfLog = false;
	mShowDebug = false;
	mApplyFlow = true;
	mSafeSpace = true;
#ifdef AN_EVALUATION_VERSION
	mEvaluation = true;
#else
	mEvaluation = false;
#endif
#else
	mStartInFullscreen = false;
	mPerfLog = false;
	mShowDebug = true;
	mApplyFlow = false;
	mSafeSpace = true;
	mEvaluation = false;
#endif
	if (mStartInFullscreen) ofToggleFullscreen();

	ofSetVerticalSync(false);

	// We need to pass the method we want ofxOpenVR to call when rending the scene
	mVive.setup(std::bind(&ofApp::render, this, std::placeholders::_1));
	mVive.setDrawControllers(true);

	ofAddListener(mVive.ofxOpenVRControllerEvent, this, &ofApp::controllerEvent);
	
	// stuff from openvr example
	{
		_texture.load("of.png");
		_texture.getTexture().setTextureWrap(GL_REPEAT, GL_REPEAT);

		_box.set(1);
		_box.enableColors();
		_box.mapTexCoordsFromTexture(_texture.getTexture());

		// Create a translation matrix to place the box in the space
		_translateMatrix.translate(ofVec3f(0.0, .0, -2.0));

		// Vertex shader source
		string vertex;

		vertex = "#version 150\n";
		vertex += STRINGIFY(
			uniform mat4 matrix;

		in vec4  position;
		in vec2  texcoord;

		out vec2 texCoordVarying;

		void main()
		{
			texCoordVarying = texcoord;
			gl_Position = matrix * position;

		}
		);

		// Fragment shader source
		string fragment = "#version 150\n";
		fragment += STRINGIFY(
			uniform sampler2DRect baseTex;

		in vec2 texCoordVarying;

		out vec4 fragColor;

		vec2 texcoord0 = texCoordVarying;

		void main() {
			vec2 tx = texcoord0;
			tx.y = 256.0 - tx.y;
			vec4 image = texture(baseTex, tx);
			fragColor = image;
		}
		);

		// Shader
		_shader.setupShaderFromSource(GL_VERTEX_SHADER, vertex);
		_shader.setupShaderFromSource(GL_FRAGMENT_SHADER, fragment);
		_shader.bindDefaults();
		_shader.linkProgram();

		// Controllers
		_controllerBox.set(.1);
		_controllerBox.enableColors();

		// Vertex shader source
		vertex = "#version 150\n";
		vertex += STRINGIFY(
			uniform mat4 matrix;

		in vec4 position;
		in vec3 v3ColorIn;

		out vec4 v4Color;

		void main() {
			v4Color.xyz = v3ColorIn; v4Color.a = 1.0;
			gl_Position = matrix * position;
		}
		);

		// Fragment shader source
		fragment = "#version 150\n";
		fragment += STRINGIFY(
			in vec4 v4Color;
		out vec4 outputColor;
		void main() {
			outputColor = v4Color;
		}
		);

		// Shader
		_controllersShader.setupShaderFromSource(GL_VERTEX_SHADER, vertex);
		_controllersShader.setupShaderFromSource(GL_FRAGMENT_SHADER, fragment);
		_controllersShader.bindDefaults();
		_controllersShader.linkProgram();

	}

#ifdef AN_USE_CREATURE		// Create instance data.
	creatureInstances.resize(2);
	for (size_t i = 0; i < creatureInstances.size(); i++) {
		creatureInstances[i].params = glm::vec3(world.urandom(), world.urandom(), world.urandom());
		glm::mat4 rot = glm::mat4_cast(glm::normalize(glm::quat((float)world.srandom(), (float)world.srandom(), (float)world.srandom(), (float)world.srandom())));
		creatureInstances[i].pos = glm::normalize(creatureInstances[i].params) * 5.f;
	}

	creatureInstances[0].params = glm::vec3(world.urandom(), world.urandom(), world.urandom());
	glm::mat4 rot = glm::mat4(1.f);
	creatureInstances[0].instanceMatrix = glm::translate(glm::mat4(), glm::vec3()) * rot;
	creatureInstances[0].instanceMatrixInverse = glm::inverse(creatureInstances[0].instanceMatrix);
#endif
#ifdef AN_USE_FISH
	fishInstances.resize(16);
	FishSegmentInstanceData& head = fishInstances[0];
	head.orient = glm::normalize(glm::quat((float)world.srandom(), (float)world.srandom(), (float)world.srandom(), (float)world.srandom()));
	//head.orient = glm::normalize(quat(vec3(M_PI/2.f, 0.f, 0.f)));
	head.position = glm::vec3(0.f, 0.f, -4.f);
	head.size = 0.5f;
	for (size_t i = 1; i < fishInstances.size(); i++) {
		FishSegmentInstanceData& parent = fishInstances[i - 1];
		FishSegmentInstanceData& child = fishInstances[i];
		glm::vec3 back = quat_uz(parent.orient) * head.size;
		child.orient = parent.orient;
		child.position = parent.position + back;
		child.size = parent.size;
	}
#endif	
/* TODO: 
#ifndef AN_FINAL
	try {
		fs::path assetpath = getAssetPath("lib.glsl").parent_path() / "*";
		wd::watch(assetpath, [this](const fs::path &path) {
			setupGPU();
		});

	}
	catch (Exception& ex) {
		cout << ex.what() << endl;
	}
#else
	setupGPU();
#endif
*/
	setupGPU();

	mFluidThread = std::thread(std::bind(&ofApp::serviceFluid, this));
	mSimulationThread = std::thread(std::bind(&ofApp::serviceSimulation, this));
	mLandscapeThread = std::thread(std::bind(&ofApp::serviceLandscape, this));
	
	/*TODO:
	// start audio processing
	auto ctx = audio::master();
	mAudioNode = ctx->makeNode(new ECAudioNode(audio::Node::Format().channels(2)));
	mAudioNode >> ctx->getOutput();
	mAudioNode->enable();
	ctx->enable();
	 */


	//shader.load(vert, frag);
	bool ok = shader.load("shaders/c.vert.glsl", "shaders/c.frag.glsl");
	printf("shader load ok %d\n", ok);

}

void ofApp::setupGPU() {

}

void ofApp::exit() {
	ofRemoveListener(mVive.ofxOpenVRControllerEvent, this, &ofApp::controllerEvent);

	mVive.exit();
}

//--------------------------------------------------------------
void ofApp::update(){

	float now = (float)ofGetElapsedTimeMillis() * 0.001;

	mTimestamp = std::time(nullptr);

	//TODO: get average framerate
	mFrameRate = ofGetFrameRate();
	// avoid infinities
	mFrameRate = MAX(100.f, MIN(mFrameRate, 1.f));
	// frame duration in seconds, hopefully ~0.010 for Vive
	mDeltaTime = 1.f / mFrameRate;

/*TODO: 
#ifdef AN_EVALUATION_VERSION
	mEvaluationNoticeTime -= mDeltaTime;
	if (mAutoScreenshotTime < 0.) {
		mEvaluation = true;
		mAutoScreenshotTime = 120.;
	}
#endif 
#ifdef AN_AUTOSCREENSHOT
	mAutoScreenshotTime -= mDeltaTime;
	if (mAutoScreenshotTime < 0.) {
		mScreenshot = true;
		mAutoScreenshotTime = 5.;
	}
#endif
*/
	// main-thread simulation of world:
	// does it matter that this is using the previous frame's nav pose? 
	// does 10ms latency in the world's *simulation data* matter? no.
	world.update_move(mDeltaTime);

	if (world.updating) {
#ifdef AN_USE_FISH
		// fish:
		FishSegmentInstanceData& head = fishInstances[0];
		head.orient = glm::normalize(head.orient * glm::quat(glm::vec3(0.1f*world.srandom(), 0.4f*world.srandom(), 0.2f*world.srandom())));
		head.position -= float(mDeltaTime) * quat_uz(head.orient) * head.size;


		for (size_t i = 1; i < fishInstances.size(); i++) {
			FishSegmentInstanceData& parent = fishInstances[i - 1];
			FishSegmentInstanceData& child = fishInstances[i];

			float tailness = (0.5f + i) / float(fishInstances.size());
			float headness = 1.f - tailness;
			float a = sin((float)(ofGetElapsedTimeMillis() * 0.001) - i);
			glm::quat qtarget = parent.orient;
			//qtarget = qtarget * quat(vec3(0.f, 0.5f*a, 0.f));



			child.orient = glm::slerp(child.orient, qtarget, 0.01f * (a + 2.f));//0.02f);


			//child.size = head.size * (1. + 0.5*sin(3.*(float)getElapsedSeconds() - i));
			child.size = parent.size * 0.97;

			glm::vec3 back = quat_uz(child.orient) * child.size;
			child.position = parent.position + back;
		}
		for (size_t i = 0; i < fishInstances.size(); i++) {
			FishSegmentInstanceData& child = fishInstances[i];
			//child.instanceMatrix = glm::translate(glm::mat4(), child.position) * mat4_cast(child.orient);
			child.instanceMatrix = glm::translate(glm::mat4(), child.position) * mat4_cast(child.orient) * glm::scale(glm::vec3(child.size));

			child.instanceMatrixInverse = glm::inverse(child.instanceMatrix);

			child.instanceMatrix1 = glm::translate(glm::mat4(), child.position) * glm::scale(glm::vec3(child.size));

			child.instanceMatrixInverse1 = glm::inverse(child.instanceMatrix1);
		}
#endif
	}


	mVive.update();
	update_nav();

	// upload content to GPU:
	// (doing this before updating nav will add 1 frame delay to the creatures, but it looks ok)
	// doing it before updating nav means our nav-to-projection-matrix latency is shorter, and that's more important.
	updateGPU();

}

void ofApp::serviceFluid() {
	printf("starting to service fluid\n");
	const double sleep_ms = 1000. / 90;
	while (!mCanceled) {

		//printf(".\n");
		const auto before = std::chrono::system_clock::now();
		if (world.updating) {
			try {
				world.fluid_update(sleep_ms * 0.001);

				// drift:
				world.fluid.velocities.front().read_interp<float>(world.pos.x, world.pos.y, world.pos.z, &mLocalFlow.x);
			}
			catch (const std::exception& ex) {
				std::cerr << ex.what() << std::endl;
			}
		}


		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - before);
		double dur = duration.count(); //ms
		//printf("fluid dur %f\n", dur);
		int tosleep = sleep_ms - dur;
		tosleep = MAX(tosleep, 0);
		if (mPerfLog) std::cout << "serviceFluid fps: " << int(1000. / dur) << "\tsleep ms:" << tosleep << std::endl;
		if (tosleep) {
			std::chrono::milliseconds mstosleep(tosleep);
			std::this_thread::sleep_for(mstosleep);
		}
	}
	std::cout << "fluid thread exited" << std::endl;
}

void ofApp::serviceSimulation() {
	printf("starting to service Simulation\n");
	const double sleep_ms = 1000. / 90;
	while (!mCanceled) {
		const auto before = std::chrono::system_clock::now();
		if (world.updating) {
			try {
				world.animate(sleep_ms);
			}
			catch (const exception& ex) {
				std::cerr << ex.what() << std::endl;
			}
		}
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - before);
		double dur = duration.count(); //ms
		//printf("fluid dur %f\n", dur);
		int tosleep = sleep_ms - dur;
		tosleep = MAX(tosleep, 0);
		if (mPerfLog) std::cout << "serviceSimulation fps: " << int(1000. / dur) << "\tsleep ms:" << tosleep << std::endl;
		if (tosleep) {
			std::chrono::milliseconds mstosleep(tosleep);
			std::this_thread::sleep_for(mstosleep);
		}
	}
	std::cout << "simulation thread exited" << std::endl;
}

void ofApp::serviceLandscape() {
	printf("starting to service Landscape\n");
	const double sleep_ms = 1000. / 90;
	while (!mCanceled) {
		const auto before = std::chrono::system_clock::now();
		if (world.updating) {
			try {
				world.landscape_update(sleep_ms);
				mLandscapeUpdated = true;
			}
			catch (const exception& ex) {
				std::cerr << ex.what() << std::endl;
			}
		}
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - before);
		double dur = duration.count(); //ms
		//printf("fluid dur %f\n", dur);
		int tosleep = sleep_ms - dur;
		tosleep = MAX(tosleep, 0);
		if (mPerfLog) std::cout << "service landscape fps: " << int(1000. / dur) << "\tsleep ms:" << tosleep << std::endl;
		if (tosleep) {
			std::chrono::milliseconds mstosleep(tosleep);
			std::this_thread::sleep_for(mstosleep);
		}
	}
	std::cout << "landscape thread exited" << std::endl;
}

void ofApp::update_controller(vr::ETrackedControllerRole nController, int i) {
	particleCollector& pc = world.collectors[i];
	Wand& wand = wands[i];
	glm::vec3 pos = glm::vec3(world.getPos());
	glm::mat4 nav = glm::mat4_cast(mNavOrient);
	glm::mat4 chap2world = glm::translate(pos) * nav;
	//printf("connected %d %d\n", i, mVive.isControllerConnected(nController));
	if (mVive.isControllerConnected(nController)) {

		glm::mat4 pose = mVive.getControllerPose(nController);
		wand.pose = chap2world * pose;

		pc.pos = glm::vec3(wand.pose[3]);
		pc.intensity = 1.f;
		//printf("hand %i: %d %d %d %d, %f %f\n", i, state.gripButton, state.menuButton, state.trackpadButton, state.triggerButton, state.trackpad.x, state.trackpad.y);
		glm::quat orient = glm::quat_cast(chap2world * pose);
		pc.orient = orient;
		if (wand.isTriggerPressed) {
			mNavVelocityWands += quat_rotate(orient, glm::vec3(0.f, 0.f, (float)(-wand.triggerPressure * mNavKeyRate.z * mNavSpeed)));
		}
		if (pc.vibrate > 0.f) {
			float period = 3000.f;
			mVive._pHMD->TriggerHapticPulse(vr::EVREye(i), 0, (unsigned short)period);
			pc.vibrate -= 0.5f;
		}


		vr::VRControllerState_t cs;
		vr::TrackedDeviceIndex_t nDevice = mVive._pHMD->GetTrackedDeviceIndexForControllerRole(nController);
		mVive._pHMD->GetControllerState(nDevice, &cs, sizeof(vr::VRControllerState_t));

		/*state.index = nDevice;
		state.menuButton = (cs.ulButtonPressed & vr::ButtonMaskFromId(vr::k_EButton_ApplicationMenu)) != 0;
		state.gripButton = (cs.ulButtonPressed & vr::ButtonMaskFromId(vr::k_EButton_Grip)) != 0;
		state.trackpadButton = (cs.ulButtonTouched & vr::ButtonMaskFromId(vr::k_EButton_SteamVR_Touchpad)) != 0;
		state.triggerButton = (cs.ulButtonTouched & vr::ButtonMaskFromId(vr::k_EButton_SteamVR_Trigger)) != 0;
		state.trackpad.x = cs.rAxis[0].x;
		state.trackpad.y = cs.rAxis[0].x;*/
		wand.triggerPressure = cs.rAxis[1].x;
		wand.isTriggerPressed = (cs.ulButtonTouched & vr::ButtonMaskFromId(vr::k_EButton_SteamVR_Trigger)) != 0;

		//printf("wand pressure %f\n", wand.triggerPressure);
	}
	else {
		pc.intensity *= 0.99f;
	}
}

void ofApp::update_nav() {
	float dt = mDeltaTime;

	// the current velocity
	glm::vec3 v;

	// start determining navigation:
	mNavOrient = world.orient;

	// apply navigation keys etc.
	// mNavVelocityKeys should be in view space
	// to put it in world space, need to unrotate from the current view direction
	// when using VR, that is the HMD direction
	// otherwise, it is the mNavOrient


	mNavVelocityWands = glm::vec3(0.f);
//#ifdef AN_USE_VR
	// navigate with wands:
	{

		update_controller(vr::TrackedControllerRole_LeftHand, 0);
		update_controller(vr::TrackedControllerRole_RightHand, 1);
		v += mNavVelocityWands;
	}
	// HMD, so rotate key nav into current facing direction:
	glm::vec3 keynav = mNavVelocityKeys;
	//keynav = quat_rotate(mNavOrient, keynav);
	//keynav = quat_unrotate(mFaceOrient, keynav);


	keynav = quat_unrotate(mFaceOrient, keynav);
	keynav = quat_rotate(mNavOrient, keynav);
	v += keynav;
//#else
	// no HMD, so rotate into current nav orient:
	//v += glm::quat_rotate(mNavOrient, mNavVelocityKeys);
//#endif
	// update CPU state for navigation:
	{
		// get navigation velocity (from wands and from keys)
		v = vec_fixnan(v);
		// increment to world position:
		glm::vec3 p(world.pos);
		p = p + v * dt;
		p = vec_fixnan(p);

		// add fluid effect:
		if (world.updating && mApplyFlow) p += vec_fixnan(mLocalFlow * (mNavSpeed * 2.f));

		// update world:
		glm::dvec3 newpos(p);
		world.dpos = vec_fixnan((newpos - world.pos) / double(dt));
		world.setPos(newpos);

		// TODO: apply  mNavAngularVelocity.y to mNavOrient
		mNavOrient *= glm::quat(mNavAngularVelocity * dt);

		world.setOrient(mNavOrient);
	}
}

void ofApp::updateGPU() {

}

//--------------------------------------------------------------
void ofApp::draw(){
	mVive.render();

	mVive.renderDistortion();

	mVive.drawDebugInfo();

}

//--------------------------------------------------------------
void  ofApp::render(vr::Hmd_Eye nEye)
{
	// OF textured cube
	glm::mat4x4 currentViewProjectionMatrix = mVive.getCurrentViewProjectionMatrix(nEye);
	glm::mat4x4 hdmPoseMat = _translateMatrix * currentViewProjectionMatrix;

	_shader.begin();
	_shader.setUniformMatrix4f("matrix", hdmPoseMat, 1);
	_shader.setUniformTexture("baseTex", _texture, 0);
	_box.draw();
	_shader.end();

	// Left controller
	if (mVive.isControllerConnected(vr::TrackedControllerRole_LeftHand)) {
		glm::mat4x4 leftControllerPoseMat = currentViewProjectionMatrix * mVive.getControllerPose(vr::TrackedControllerRole_LeftHand);

		_controllersShader.begin();
		_controllersShader.setUniformMatrix4f("matrix", leftControllerPoseMat, 1);
		_controllerBox.drawWireframe();
		_controllersShader.end();
	}

	// Right controller
	if (mVive.isControllerConnected(vr::TrackedControllerRole_RightHand)) {
		glm::mat4x4 rightControllerPoseMat = currentViewProjectionMatrix * mVive.getControllerPose(vr::TrackedControllerRole_RightHand);

		_controllersShader.begin();
		_controllersShader.setUniformMatrix4f("matrix", rightControllerPoseMat, 1);
		_controllerBox.drawWireframe();
		_controllersShader.end();
	}
}

void ofApp::controllerEvent(ofxOpenVRControllerEventArgs& args)
{
	cout << "ofApp::controllerEvent > role: " << (int)args.controllerRole << " - event type: " << (int)args.eventType << " - button type: " << (int)args.buttonType << " - x: " << args.analogInput_xAxis << " - y: " << args.analogInput_yAxis << endl;
	if (args.eventType == EventType::ButtonPress || args.eventType == EventType::ButtonTouch){
		mSafeSpace = false;
		mEvaluation = false;
	}


}
//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	mVive.toggleGrid();
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
