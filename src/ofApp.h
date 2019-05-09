#pragma once

#include "ofMain.h"
#include "World.h"
#include "ofxOpenVR.h"

#define AN_USE_CREATURE
#define AN_USE_FISH
//#define AN_FINAL

struct Wand {
	glm::mat4 pose;
	bool isTriggerPressed;
	float triggerPressure;
};

struct FishSegmentInstanceData {
	glm::mat4 instanceMatrix, instanceMatrixInverse;
	glm::mat4 instanceMatrix1, instanceMatrixInverse1;
	glm::quat orient;
	glm::vec3 position;
	float size;
};

struct HandInstanceData {
	glm::mat4 instanceMatrix, instanceMatrixInverse;
};

struct CreatureInstanceData {
	glm::mat4 instanceMatrix, instanceMatrixInverse;
	glm::vec3 pos;
	glm::vec3 params;
};

struct OrganismInstanceData {
	glm::vec4 orient;
	glm::vec3 position;
	glm::vec3 params;
};

struct StalkInstanceData {
	glm::vec4 orient, orient2;
	glm::vec2 size, size2;
	glm::vec3 color, color2;
	glm::vec3 position;
	float type; // root, branch, leaf
};

class ofApp : public ofBaseApp{
public:
	
	World& world;



	ofxOpenVR mVive;
	Wand wands[2];

	//Stuff from OpenVR example
	ofImage _texture;
	ofBoxPrimitive _box;
	ofMatrix4x4 _translateMatrix;
	ofShader _shader;
	ofBoxPrimitive _controllerBox;
	ofShader _controllersShader;

	/*gl::Texture3dRef mLandTex1, mLandTex2, mNoiseTex;
	gl::Texture2dRef mSafeSpaceTex, mEvaluationTex;
	gl::VboRef mParticleVbo, mOrganismInstanceDataVbo, mStalkInstanceDataVbo, mCreatureInstanceDataVbo, mHandInstanceDataVbo, mFishInstanceDataVbo;
	gl::GlslProgRef mLandShader, mParticleShader, mOrganismShader, mStalkShader, mCreatureShader, mHandShader, mFishShader;
	gl::BatchRef mParticleBatch, mGhostBatch, mLandBatch, mOrganismBatch, mStalkBatch, mMirrorBatch, mCreatureBatch, mHandBatch, mFishBatch;
	*/
	std::vector<OrganismInstanceData> organismInstances;
	std::vector<StalkInstanceData> stalkInstances;
	std::vector<HandInstanceData> handInstances;

	std::vector<CreatureInstanceData> creatureInstances;
	std::vector<FishSegmentInstanceData> fishInstances;

	int mOrganismInstanceCount, mStalkInstanceCount;

	glm::quat mNavOrient; // the transform from real space to world space
	glm::quat mFaceOrient; // the orientation of the HMD in real space
	glm::vec3 mNavVelocityKeys; // in view space
	glm::vec3 mNavVelocityWands; // in world space
	glm::vec3 mNavAngularVelocity;
	glm::vec3 mNavKeyRate, mNavKeyTurnRate;
	glm::vec3 mLocalFlow;
	float mNavSpeed;

	glm::mat4 mRenderProjectionMatrix, mRenderViewMatrix;

	float mFrameRate, mDeltaTime;
	float mSleepMinMS;
	float mLandtexmix;
	int mLandtexmixTarget;
	int mTimestamp;

	double mAutoScreenshotTime, mEvaluationNoticeTime;

	bool mStartInFullscreen, mPerfLog = true, mShowDebug;
	bool mApplyFlow;
	bool mShowWorld, mShowLanscape, mShowOrganisms, mShowParticles, mShowStalks, mShowCreatures, mShowControllers;

	bool mCanceled, mLandscapeUpdated;
	bool mScreenshot;

	bool mSafeSpace, mEvaluation;

	std::thread mSimulationThread, mLandscapeThread, mFluidThread;
	//ECAudioNodeRef mAudioNode;


	ofShader shader;
	ofPlanePrimitive plane;

	ofApp() : world(World::get()) {
		printf("created world\n");}

	void setup();
	void setupGPU();
	void exit();
	void update();
	void serviceFluid();
	void serviceSimulation();
	void serviceLandscape();
	void update_nav();
	void updateGPU();
	void draw();

	void keyPressed(int key);
	void keyReleased(int key);
	void mouseMoved(int x, int y );
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y);
	void mouseExited(int x, int y);
	void windowResized(int w, int h);
	void dragEvent(ofDragInfo dragInfo);
	void gotMessage(ofMessage msg);
	
	void render(vr::Hmd_Eye nEye);
	void controllerEvent(ofxOpenVRControllerEventArgs& args);
	void update_controller(vr::ETrackedControllerRole nController, int i);

};
