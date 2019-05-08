
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/audio/Voice.h"
#include "cinder/audio/Context.h"
#include "cinder/audio/Node.h"
#include "cinder/audio/GenNode.h"
#include "cinder/audio/MonitorNode.h"
#include "cinder/audio/GainNode.h"
#include "cinder/audio/Device.h"
#include "cinder/params/Params.h"
#include "cinder/GeomIO.h"
#include "cinder/Json.h"
#include "cinder/System.h"
#include "cinder/Utilities.h"
#include "Watchdog.h"

#include "Resources.h"

#include "glm/gtc/quaternion.hpp"

#include "World.h"

#include <algorithm>

using namespace ci;
using namespace ci::app;
using namespace std;


//#define AN_EVALUATION_VERSION 1

#ifdef _MSC_VER

// Windows:

// build for VR:
#define AN_USE_VR 1
#ifdef AN_USE_VR
#include "CinderVive.h"
#endif

// build for distribution:
#define AN_FINAL 1

#ifdef AN_FINAL

#else
//#define AN_AUTOSCREENSHOT 1
#endif

#include <Windows.h>
#include <io.h>
#include <Fcntl.h>
#ifndef AN_FINAL
	#define AN_USE_CONSOLE 1
#endif
#else 

// OSX:
#define AN_USE_FISH 1

#endif

//#define AN_USE_CREATURE 1

struct FishSegmentInstanceData {
	mat4 instanceMatrix, instanceMatrixInverse;
	mat4 instanceMatrix1, instanceMatrixInverse1;
	quat orient;
	vec3 position;
	float size;
};

struct HandInstanceData {
	mat4 instanceMatrix, instanceMatrixInverse;
};

struct CreatureInstanceData {
	mat4 instanceMatrix, instanceMatrixInverse;
	vec3 pos;
	vec3 params;
};

struct OrganismInstanceData {
	vec4 orient;
	vec3 position;
	vec3 params;
};

struct StalkInstanceData {
	vec4 orient, orient2;
	vec2 size, size2;
	vec3 color, color2;
	vec3 position;
	float type; // root, branch, leaf
};

class ECAudioNode : public ci::audio::Node {
public:

	ECAudioNode(const Format &format = Format()) : Node(format), world(World::get()) {}

protected:

	void initialize() override {
		world.dsp_initialize(
			(double)audio::master()->getOutput()->getOutputSampleRate(),
			(long)audio::master()->getOutput()->getFramesPerBlock()
			);
	}

	void process(ci::audio::Buffer *buffer) override {
		float * L = buffer->getChannel(0);
		float * R = buffer->getChannel(1);
		world.performStereo(L, R, (long)buffer->getNumFrames());
	}

private:
	World& world;
};
typedef std::shared_ptr<class ECAudioNode>	ECAudioNodeRef;

class EndlessCurrentVRApp : public App {
public:

	World& world;

#ifdef AN_USE_VR
	hmd::HtcViveRef		mVive;
#endif
	gl::Texture3dRef mLandTex1, mLandTex2, mNoiseTex;
	gl::Texture2dRef mSafeSpaceTex, mEvaluationTex;
	gl::VboRef mParticleVbo, mOrganismInstanceDataVbo, mStalkInstanceDataVbo, mCreatureInstanceDataVbo, mHandInstanceDataVbo, mFishInstanceDataVbo;
	gl::GlslProgRef mLandShader, mParticleShader, mOrganismShader, mStalkShader, mCreatureShader, mHandShader, mFishShader;
	gl::BatchRef mParticleBatch, mGhostBatch, mLandBatch, mOrganismBatch, mStalkBatch, mMirrorBatch, mCreatureBatch, mHandBatch, mFishBatch;

	std::vector<OrganismInstanceData> organismInstances;
	std::vector<StalkInstanceData> stalkInstances;
	std::vector<HandInstanceData> handInstances;
#ifdef AN_USE_CREATURE
	std::vector<CreatureInstanceData> creatureInstances;
#endif
#ifdef AN_USE_FISH
	std::vector<FishSegmentInstanceData> fishInstances;
#endif
	int mOrganismInstanceCount, mStalkInstanceCount;

	quat mNavOrient; // the transform from real space to world space
	quat mFaceOrient; // the orientation of the HMD in real space
	vec3 mNavVelocityKeys; // in view space
	vec3 mNavVelocityWands; // in world space
	vec3 mNavAngularVelocity;
	vec3 mNavKeyRate, mNavKeyTurnRate;
	vec3 mLocalFlow;
	float mNavSpeed;

	mat4 mRenderProjectionMatrix, mRenderViewMatrix;

	float mFrameRate, mDeltaTime; 
	float mSleepMinMS;
	float mLandtexmix;
	int mLandtexmixTarget;
	int mTimestamp;

	double mAutoScreenshotTime, mEvaluationNoticeTime;

	bool mStartInFullscreen, mPerfLog, mShowDebug;
	bool mApplyFlow;
	bool mShowWorld, mShowLanscape, mShowOrganisms, mShowParticles, mShowStalks, mShowCreatures, mShowControllers;

	bool mCanceled, mLandscapeUpdated;
	bool mScreenshot;

	bool mSafeSpace, mEvaluation;

	Font mFont;
	std::thread mSimulationThread, mLandscapeThread, mFluidThread;
	ECAudioNodeRef mAudioNode;

	EndlessCurrentVRApp() : world(World::get()) {	}

	void setup() override {
		
		auto datapath = getAppPath().concat(std::string("data"));
		if (fs::exists(datapath)) addAssetDirectory(datapath);
		
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

		mFaceOrient = mNavOrient = quat_cast(mat4(1.0));


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
		mFont = Font("Helvetica", 32.0f);

		setFrameRate(90.0f);
		if (mStartInFullscreen) goFullScreen();
#ifdef AN_USE_VR
		try {
			mVive = hmd::HtcVive::create();
		}
		catch (const hmd::ViveExeption& exc) {
			CI_LOG_E(exc.what());
		}
#endif
		

#ifdef AN_USE_CREATURE		// Create instance data.
		creatureInstances.resize(2);
		for (size_t i = 0; i < creatureInstances.size(); i++) {
			creatureInstances[i].params = vec3(world.urandom(), world.urandom(), world.urandom());
			mat4 rot = mat4_cast(glm::normalize(quat((float)world.srandom(), (float)world.srandom(), (float)world.srandom(), (float)world.srandom())));
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
		head.orient = glm::normalize(quat((float)world.srandom(), (float)world.srandom(), (float)world.srandom(), (float)world.srandom()));
		//head.orient = glm::normalize(quat(vec3(M_PI/2.f, 0.f, 0.f)));
		head.position = vec3(0.f, 0.f, -4.f);
		head.size = 0.5f;
		for (size_t i = 1; i < fishInstances.size(); i++) {
			FishSegmentInstanceData& parent = fishInstances[i-1];
			FishSegmentInstanceData& child = fishInstances[i];
			vec3 back = quat_uz(parent.orient) * head.size;
			child.orient = parent.orient;
			child.position = parent.position + back;
			child.size = parent.size;
		}
#endif	
		
		// initialize GPU:
#ifndef AN_FINAL
		try {
			fs::path assetpath = getAssetPath("lib.glsl").parent_path() / "*";
			wd::watch(assetpath, [this]( const fs::path &path) {
				setupGPU();
			});
			
		} catch (Exception& ex) {
			cout << ex.what() << endl;
		}
#else
		setupGPU();
#endif
			
		mFluidThread = thread(bind(&EndlessCurrentVRApp::serviceFluid, this));
		mSimulationThread = thread(bind(&EndlessCurrentVRApp::serviceSimulation, this));
		mLandscapeThread = thread(bind(&EndlessCurrentVRApp::serviceLandscape, this));

		// start audio processing
		auto ctx = audio::master();
		mAudioNode = ctx->makeNode(new ECAudioNode(audio::Node::Format().channels(2)));
		mAudioNode >> ctx->getOutput(); 
		mAudioNode->enable();
		ctx->enable();
	}

	void loadShader(const ci::fs::path &vertexRelPath, const ci::fs::path &fragmentRelPath, DataSourceRef vert, DataSourceRef frag, std::function<void(ci::DataSourceRef, ci::DataSourceRef)> callback) {
#ifdef AN_FINAL
		callback(vert, frag);
#else
		// TODO: turn this into an auto-loading asset manager... 

		if (ci::fs::exists(ci::app::getAssetPath(vertexRelPath)) && ci::fs::exists(ci::app::getAssetPath(fragmentRelPath))) {
			callback(ci::app::loadAsset(vertexRelPath), ci::app::loadAsset(fragmentRelPath));
		}
		else {
			throw ci::app::AssetLoadExc(vertexRelPath); // not necessary correct!
		}
#endif
	}

	void setupGPU() {

		gl::clearColor(Color::black());
		gl::enableVerticalSync(false);
		gl::lineWidth(1.0f);

		loadResource(w_frag_glsl);

		try {
			loadShader("w.vert.glsl", "w.frag.glsl", loadResource(w_vert_glsl), loadResource(w_frag_glsl), [this](DataSourceRef vert, DataSourceRef frag) {
				try {
					mLandShader = gl::GlslProg::create(vert, frag);
					mLandShader->uniform("landtex1", 0);
					mLandShader->uniform("landtex2", 1);
					mLandShader->uniform("noisetex", 2);
					mLandShader->uniform("dim", float(DIM));
					mLandBatch = gl::Batch::create(geom::Plane(), mLandShader);
				}
				catch (gl::GlslProgCompileExc exc) { cout << exc.what() << endl; }
			});

			gl::Texture3d::Format format3d;
			// format3d.maxAnisotropy( ? )
			format3d.internalFormat(GL_RGBA32F) // GL_RGBA8_SNORM GL_RGBA32F
				.wrap(GL_REPEAT)
				.mipmap(true) // ?
				.minFilter(GL_LINEAR)
				.magFilter(GL_LINEAR)
				.label("landscape3D");
			format3d.setDataType(GL_FLOAT);

			mLandTex1 = gl::Texture3d::create(DIM, DIM, DIM, format3d);
			mLandTex1->update(world.landscape.ptr(), GL_RGBA,
				format3d.getDataType(),
				0, mLandTex1->getWidth(),
				mLandTex1->getHeight(),
				mLandTex1->getDepth());
			
			mLandTex2 = gl::Texture3d::create(DIM, DIM, DIM, format3d);
			mLandTex2->update(world.landscape.ptr(), GL_RGBA,
							 format3d.getDataType(),
							 0, mLandTex2->getWidth(),
							 mLandTex2->getHeight(),
							 mLandTex2->getDepth());
			
			mNoiseTex = gl::Texture3d::create(DIM, DIM, DIM, format3d);
			mNoiseTex->update(world.noisefield.ptr(), GL_RGBA,
				format3d.getDataType(),
				0, mNoiseTex->getWidth(),
				mNoiseTex->getHeight(),
				mNoiseTex->getDepth());

			loadShader("p.vert.glsl", "p.frag.glsl", loadResource(p_vert_glsl), loadResource(p_frag_glsl), [this](DataSourceRef vert, DataSourceRef frag) {
				try {

					mParticleShader = gl::GlslProg::create(vert, frag);

					// Create particle buffer on GPU and copy over data.
					// Mark as streaming, since we will copy new data every frame.
					mParticleVbo = gl::Vbo::create(GL_ARRAY_BUFFER,
						sizeof(particle_base) * NUM_PARTICLES,
						world.particle_bases,
						GL_STREAM_DRAW);

					// Describe particle semantics for GPU.
					geom::BufferLayout particleLayout;
					particleLayout.append(geom::Attrib::COLOR, 4, sizeof(particle_base), offsetof(particle_base, color));
					particleLayout.append(geom::Attrib::POSITION, 3, sizeof(particle_base), offsetof(particle_base, pos));

					// Create mesh by pairing our particle layout with our particle Vbo.
					// A VboMesh is an array of layout + vbo pairs
					auto mesh = gl::VboMesh::create(NUM_PARTICLES, GL_POINTS, { { particleLayout, mParticleVbo } });

					mParticleBatch = gl::Batch::create(mesh, mParticleShader);

				}
				catch (gl::GlslProgCompileExc exc) { cout << exc.what() << endl; }
			});

			loadShader("s.vert.glsl", "s.frag.glsl", loadResource(s_vert_glsl), loadResource(s_frag_glsl), [this](DataSourceRef vert, DataSourceRef frag) {
				try {
					mStalkShader = gl::GlslProg::create(vert, frag);

					// this seems a bit long-winded.
					gl::VboRef stalkVbo = gl::Vbo::create(GL_ARRAY_BUFFER,
						NUM_STALK_VERTICES * sizeof(vertex),
						world.stalkMesh,
						GL_STATIC_DRAW);

					geom::BufferLayout stalkDataLayout;
					stalkDataLayout.append(geom::Attrib::POSITION, 3, sizeof(vertex), offsetof(vertex, pos), 0);
					stalkDataLayout.append(geom::Attrib::NORMAL, 3, sizeof(vertex), offsetof(vertex, normal), 0);

					std::vector<std::pair<geom::BufferLayout, gl::VboRef>> vertexArrayBuffers;
					vertexArrayBuffers.push_back(std::pair<geom::BufferLayout, gl::VboRef>(stalkDataLayout, stalkVbo));

					gl::VboMeshRef m = gl::VboMesh::create(geom::Cylinder().subdivisionsAxis(18).subdivisionsHeight(2).set(vec3(0), vec3(0, 0, 1)));
					//gl::VboMeshRef m = gl::VboMesh::create(NUM_STALK_VERTICES, GL_QUADS, vertexArrayBuffers);

					// Create instance data.
					stalkInstances.resize(NUM_STALKS);

					mStalkInstanceDataVbo = gl::Vbo::create(GL_ARRAY_BUFFER,
						stalkInstances.size() * sizeof(StalkInstanceData),
						stalkInstances.data(),
						GL_DYNAMIC_DRAW);

					geom::BufferLayout instanceDataLayout;
					instanceDataLayout.append(geom::Attrib::CUSTOM_0, 4, sizeof(StalkInstanceData), offsetof(StalkInstanceData, orient), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_1, 4, sizeof(StalkInstanceData), offsetof(StalkInstanceData, orient2), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_2, 2, sizeof(StalkInstanceData), offsetof(StalkInstanceData, size), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_3, 2, sizeof(StalkInstanceData), offsetof(StalkInstanceData, size2), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_4, 3, sizeof(StalkInstanceData), offsetof(StalkInstanceData, color), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_5, 3, sizeof(StalkInstanceData), offsetof(StalkInstanceData, color2), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_6, 3, sizeof(StalkInstanceData), offsetof(StalkInstanceData, position), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_7, 1, sizeof(StalkInstanceData), offsetof(StalkInstanceData, type), 1);

					// Add our instance data buffer to the model:
					m->appendVbo(instanceDataLayout, mStalkInstanceDataVbo);

					mStalkBatch = gl::Batch::create(m, mStalkShader, {
						{ geom::Attrib::CUSTOM_0, "vInstanceOrientation" },
						{ geom::Attrib::CUSTOM_1, "vInstanceOrientation2" },
						{ geom::Attrib::CUSTOM_2, "vInstanceSize" },
						{ geom::Attrib::CUSTOM_3, "vInstanceSize2" },
						{ geom::Attrib::CUSTOM_4, "vInstanceColor" },
						{ geom::Attrib::CUSTOM_5, "vInstanceColor2" },
						{ geom::Attrib::CUSTOM_6, "vInstancePosition" },
						{ geom::Attrib::CUSTOM_7, "vInstanceType" },
					});

				}
				catch (gl::GlslProgCompileExc exc) { cout << exc.what() << endl; }
			});
			
#ifdef AN_USE_FISH
			loadShader("f.vert.glsl", "f.frag.glsl", loadResource(f_vert_glsl), loadResource(f_frag_glsl), [this](DataSourceRef vert, DataSourceRef frag) {
				try {
					mFishShader = gl::GlslProg::create(vert, frag);
					
					gl::VboMeshRef m = gl::VboMesh::create(geom::Cube());
					
					mFishInstanceDataVbo = gl::Vbo::create(GL_ARRAY_BUFFER,
															   fishInstances.size() * sizeof(FishSegmentInstanceData),
															   fishInstances.data(),
															   GL_DYNAMIC_DRAW);
					
					geom::BufferLayout instanceDataLayout;
					instanceDataLayout.append(geom::Attrib::CUSTOM_0, 16, sizeof(FishSegmentInstanceData), offsetof(FishSegmentInstanceData, instanceMatrix), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_1, 16, sizeof(FishSegmentInstanceData), offsetof(FishSegmentInstanceData, instanceMatrixInverse), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_2, 16, sizeof(FishSegmentInstanceData), offsetof(FishSegmentInstanceData, instanceMatrix1), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_3, 16, sizeof(FishSegmentInstanceData), offsetof(FishSegmentInstanceData, instanceMatrixInverse1), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_4, 4, sizeof(FishSegmentInstanceData), offsetof(FishSegmentInstanceData, orient), 1);
					
					// Add our instance data buffer to the model:
					m->appendVbo(instanceDataLayout, mFishInstanceDataVbo);
					
					mFishBatch = gl::Batch::create(m, mFishShader, {
						{ geom::Attrib::CUSTOM_0, "vInstanceMatrix" },
						{ geom::Attrib::CUSTOM_1, "vInstanceMatrixInverse" },
						{ geom::Attrib::CUSTOM_2, "vInstanceMatrix1" },
						{ geom::Attrib::CUSTOM_3, "vInstanceMatrixInverse1" },
						{ geom::Attrib::CUSTOM_4, "vInstanceOrient" },
					});

				}
				catch (gl::GlslProgCompileExc exc) { cout << exc.what() << endl; }
			});
#endif

			loadShader("o.vert.glsl", "o.frag.glsl", loadResource(o_vert_glsl), loadResource(o_frag_glsl), [this](DataSourceRef vert, DataSourceRef frag) {
				try {
					mOrganismShader = gl::GlslProg::create(vert, frag);

					// this seems a bit long-winded.
					gl::VboRef organismVbo = gl::Vbo::create(GL_ARRAY_BUFFER,
						NUM_ORGANISM_VERTICES * sizeof(vertex),
						world.organismMesh,
						GL_STATIC_DRAW);

					geom::BufferLayout organismDataLayout;
					organismDataLayout.append(geom::Attrib::POSITION, 3, sizeof(vertex), offsetof(vertex, pos), 0);
					organismDataLayout.append(geom::Attrib::NORMAL, 3, sizeof(vertex), offsetof(vertex, normal), 0);

					std::vector<std::pair<geom::BufferLayout, gl::VboRef>> vertexArrayBuffers;
					vertexArrayBuffers.push_back(std::pair<geom::BufferLayout, gl::VboRef>(organismDataLayout, organismVbo));

					gl::VboMeshRef m = gl::VboMesh::create(NUM_ORGANISM_VERTICES, GL_TRIANGLE_FAN, vertexArrayBuffers);

					// Create instance data.
					organismInstances.resize(NUM_ORGANISMS);

					mOrganismInstanceDataVbo = gl::Vbo::create(GL_ARRAY_BUFFER,
						organismInstances.size() * sizeof(OrganismInstanceData),
						organismInstances.data(),
						GL_DYNAMIC_DRAW);

					geom::BufferLayout instanceDataLayout;
					instanceDataLayout.append(geom::Attrib::CUSTOM_0, 3, sizeof(OrganismInstanceData), offsetof(OrganismInstanceData, position), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_1, 4, sizeof(OrganismInstanceData), offsetof(OrganismInstanceData, orient), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_2, 3, sizeof(OrganismInstanceData), offsetof(OrganismInstanceData, params), 1);

					// Add our instance data buffer to the model:
					m->appendVbo(instanceDataLayout, mOrganismInstanceDataVbo);

					mOrganismBatch = gl::Batch::create(m, mOrganismShader, {
						{ geom::Attrib::CUSTOM_0, "vInstancePosition" },
						{ geom::Attrib::CUSTOM_1, "vInstanceOrientation" },
						{ geom::Attrib::CUSTOM_2, "vInstanceParams" },
					});

				}
				catch (gl::GlslProgCompileExc exc) { cout << exc.what() << endl; }
			});
#ifdef AN_USE_CREATURE
			loadShader("c.vert.glsl", "c.frag.glsl", loadResource(c_vert_glsl), loadResource(c_frag_glsl), [this](DataSourceRef vert, DataSourceRef frag) {
				try {
					mCreatureShader = gl::GlslProg::create(vert, frag);
					gl::VboMeshRef m = gl::VboMesh::create(geom::Cube().size(2, 2, 2));

					mCreatureInstanceDataVbo = gl::Vbo::create(GL_ARRAY_BUFFER,
						creatureInstances.size() * sizeof(CreatureInstanceData),
						creatureInstances.data(),
						GL_DYNAMIC_DRAW);

					geom::BufferLayout instanceDataLayout;
					instanceDataLayout.append(geom::Attrib::CUSTOM_0, 16, sizeof(CreatureInstanceData), offsetof(CreatureInstanceData, instanceMatrix), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_1, 16, sizeof(CreatureInstanceData), offsetof(CreatureInstanceData, instanceMatrixInverse), 1);
					instanceDataLayout.append(geom::Attrib::CUSTOM_2, 3, sizeof(CreatureInstanceData), offsetof(CreatureInstanceData, params), 1);

					// Add our instance data buffer to the model:
					m->appendVbo(instanceDataLayout, mCreatureInstanceDataVbo);

					mCreatureBatch = gl::Batch::create(m, mCreatureShader, {
						{ geom::Attrib::CUSTOM_0, "vInstanceMatrix" },
						{ geom::Attrib::CUSTOM_1, "vInstanceMatrixInverse" },
						{ geom::Attrib::CUSTOM_2, "vInstanceParams" },
					});

				}
				catch (gl::GlslProgCompileExc exc) { cout << exc.what() << endl; }
			});
#endif
			loadShader("h.vert.glsl", "h.frag.glsl", loadResource(h_vert_glsl), loadResource(h_frag_glsl), [this](DataSourceRef vert, DataSourceRef frag) {
				try {
					mHandShader = gl::GlslProg::create(vert, frag);
					gl::VboMeshRef m = gl::VboMesh::create(geom::Cube().size(1, 1, 1));

					mHandBatch = gl::Batch::create(m, mHandShader);

				}
				catch (gl::GlslProgCompileExc exc) { cout << exc.what() << endl; }
			});
		}
		catch (Exception& ex) {
			cout << ex.what() << endl;
		}
	}

	void cleanup() override {

		world.updating = false;
		mCanceled = true;
		mFluidThread.join();
		mSimulationThread.join();
		mLandscapeThread.join();
	}

	void goFullScreen() {
		bool fs = !isFullScreen();
		setFullScreen(fs);

#ifdef _MSC_VER
		if (!fs) {
			while (ShowCursor(TRUE) < 0);
		}
		else {
			while (ShowCursor(FALSE) >= 0);
		}
#else
		if (fs) {
			hideCursor();
		}
		else {
			showCursor();
		}
#endif
	}

	void serviceFluid() {
		ThreadSetup threadSetup;
		const double sleep_s = 1. / getFrameRate();
		while (!mCanceled) {
			Timer t(true);
			if (world.updating) {
				try {
					world.fluid_update(sleep_s);

					// drift:
					world.fluid.velocities.front().read_interp<float>(world.pos.x, world.pos.y, world.pos.z, &mLocalFlow.x);
				}
				catch (const Exception& ex) {
					std::cerr << ex.what() << std::endl;
				}
			}



			double dur = t.getSeconds();
			double tosleep = sleep_s - dur;
			tosleep = MAX(tosleep, 0.);
			if (mPerfLog) std::cout << "serviceFluid: " << int(1. / dur) << "\tsleep:" << tosleep << std::endl;
			ci::sleep(mSleepMinMS + (float)tosleep * 1000.f);
		}
		std::cout << "fluid thread exited" << std::endl;
	};

	void serviceSimulation() {
		ThreadSetup threadSetup;
		const double sleep_s = 1. / getFrameRate();
		while (!mCanceled) {
			Timer t(true);
			if (world.updating) {
				try {
					world.animate(sleep_s);
				}
				catch (const Exception& ex) {
					std::cerr << ex.what() << std::endl;
				}
			}
			double dur = t.getSeconds();
			double tosleep = sleep_s - dur;
			tosleep = MAX(tosleep, 0.);
			if (mPerfLog) std::cout << "serviceSimulation: " << int(1. / dur) << "\tsleep:" << tosleep << std::endl;
			ci::sleep(mSleepMinMS + (float)tosleep * 1000.f);
		}
		std::cout << "simulation thread exited" << std::endl;
	};

	void serviceLandscape() {
		ThreadSetup threadSetup;
		const double sleep_s = 1. / getFrameRate();
		while (!mCanceled) {
			Timer t(true);
			if (world.updating) {
				try {
					world.landscape_update(sleep_s);
					mLandscapeUpdated = true;
				}
				catch (const Exception& ex) {
					std::cerr << ex.what() << std::endl;
				}
			}
			double dur = t.getSeconds();
			float tosleep = float( sleep_s - dur );
			tosleep = MAX(tosleep, 0.f);
			if (mPerfLog) std::cout << "serviceLandscape FPS: " << int(1. / dur) << "\tsleep:" << tosleep << std::endl;
			ci::sleep(mSleepMinMS + (float)tosleep * 1000.f);
		}
		std::cout << "landscape thread exited" << std::endl;
	};

	void update() override {
		
		float now = (float)getElapsedSeconds();
		
		mTimestamp = std::time(nullptr);

		mFrameRate = getAverageFps();
		// avoid infinities
		mFrameRate = MAX(100.f, MIN(mFrameRate, 1.f));
		// frame duration in seconds, hopefully ~0.010 for Vive
		mDeltaTime = 1.f / mFrameRate;

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
		// main-thread simulation of world:
		// does it matter that this is using the previous frame's nav pose? 
		// does 10ms latency in the world's *simulation data* matter? no.
		world.update_move(mDeltaTime);
		
		if (world.updating) {
#ifdef AN_USE_FISH
			// fish:
			FishSegmentInstanceData& head = fishInstances[0];
			head.orient = glm::normalize(head.orient * quat(vec3(0.1f*world.srandom(), 0.4f*world.srandom(), 0.2f*world.srandom())));
			head.position -= float(mDeltaTime) * quat_uz(head.orient) * head.size;
			
			
			for (size_t i = 1; i < fishInstances.size(); i++) {
				FishSegmentInstanceData& parent = fishInstances[i-1];
				FishSegmentInstanceData& child = fishInstances[i];
				
				float tailness = (0.5f+i)/float(fishInstances.size());
				float headness = 1.f-tailness;
				float a = sin((float)getElapsedSeconds() - i);
				quat qtarget = parent.orient;
				//qtarget = qtarget * quat(vec3(0.f, 0.5f*a, 0.f));
				
				
				
				child.orient = slerp(child.orient, qtarget, 0.01f * (a+2.f));//0.02f);
				
				
				//child.size = head.size * (1. + 0.5*sin(3.*(float)getElapsedSeconds() - i));
				child.size = parent.size * 0.97;
				
				vec3 back = quat_uz(child.orient) * child.size;
				child.position = parent.position + back;
			}
			for (size_t i = 0; i < fishInstances.size(); i++) {
				FishSegmentInstanceData& child = fishInstances[i];
				//child.instanceMatrix = glm::translate(glm::mat4(), child.position) * mat4_cast(child.orient);
				child.instanceMatrix = glm::translate(glm::mat4(), child.position) * mat4_cast(child.orient) * glm::scale(vec3(child.size));
				
				child.instanceMatrixInverse = glm::inverse(child.instanceMatrix);
				
				child.instanceMatrix1 = glm::translate(glm::mat4(), child.position) * glm::scale(vec3(child.size));
				
				child.instanceMatrixInverse1 = glm::inverse(child.instanceMatrix1);
			}
#endif
		}
		
		// upload content to GPU:
		// (doing this before updating nav will add 1 frame delay to the creatures, but it looks ok)
		// doing it before updating nav means our nav-to-projection-matrix latency is shorter, and that's more important.
		updateGPU();
	}

	void update_nav() {
		float dt = mDeltaTime;

		// the current velocity
		vec3 v;

		// start determining navigation:
		mNavOrient = world.orient;

		// apply navigation keys etc.
		// mNavVelocityKeys should be in view space
		// to put it in world space, need to unrotate from the current view direction
		// when using VR, that is the HMD direction
		// otherwise, it is the mNavOrient


		mNavVelocityWands = glm::vec3(0.f);
#ifdef AN_USE_VR
		// navigate with wands:
		if (mVive) {
			mVive->update();

			vec3 pos = vec3(world.getPos());
			mat4 nav = glm::mat4_cast(mNavOrient);
			mat4 chap2world = glm::translate(pos) * nav;

			for (int i = 0; i < 2; i++) {

				hmd::HandControllerState state = mVive->getHandController(vr::EVREye(i));
				particleCollector& pc = world.collectors[i];

				if (state.isValid) {
					if ((state.trigger > 0.25f || state.menuButton || state.gripButton || state.trackpadButton || state.triggerButton)) {
						mSafeSpace = false;
						mEvaluation = false;
					}


					pc.pos = glm::vec3((chap2world * state.pose)[3]);
					pc.intensity = 1.f;
					//printf("hand %i: %d %d %d %d, %f %f\n", i, state.gripButton, state.menuButton, state.trackpadButton, state.triggerButton, state.trackpad.x, state.trackpad.y);
					quat orient = glm::quat_cast(chap2world * state.pose);
					pc.orient = orient;
					if (state.triggerButton) {
						mNavVelocityWands += quat_rotate(orient, vec3(0.f, 0.f, (float)(-state.trigger * mNavKeyRate.z * mNavSpeed)));
					}
					if (pc.vibrate > 0.f) {
						float period = 3000.f;
						mVive->triggerHapticPulse(vr::EVREye(i), (unsigned short)period);
						pc.vibrate -= 0.5f;
					}
				}
				else {
					pc.intensity *= 0.99f;
				}
			}
			v += mNavVelocityWands;
		}
		// HMD, so rotate key nav into current facing direction:
		vec3 keynav = mNavVelocityKeys;
		//keynav = quat_rotate(mNavOrient, keynav);
		//keynav = quat_unrotate(mFaceOrient, keynav);


		keynav = quat_unrotate(mFaceOrient, keynav);
		keynav = quat_rotate(mNavOrient, keynav);
		v += keynav;
#else
		// no HMD, so rotate into current nav orient:
		v += quat_rotate(mNavOrient, mNavVelocityKeys);
#endif
		// update CPU state for navigation:
		{
			// get navigation velocity (from wands and from keys)
			v = vec_fixnan(v);
			// increment to world position:
			vec3 p(world.pos);
			p = p + v*dt;
			p = vec_fixnan(p);

			// add fluid effect:
			if (world.updating && mApplyFlow) p += vec_fixnan(mLocalFlow * (mNavSpeed * 2.f));

			// update world:
			dvec3 newpos(p);
			world.dpos = vec_fixnan((newpos - world.pos) / double(dt));
			world.setPos(newpos);

			// TODO: apply  mNavAngularVelocity.y to mNavOrient
			mNavOrient *= quat(mNavAngularVelocity * dt);

			world.setOrient(mNavOrient);
		}
	}

	void updateGPU() {

		vec2 textsize(256, 256);
		{
			string txt = "press any key or button to begin...";
			TextBox tbox = TextBox()
				.alignment(TextBox::CENTER)
				.font(mFont)
				.size(textsize)
				.text(txt);
			tbox.setColor(Color(1.0f, 1.f, 1.f));
			tbox.setBackgroundColor(ColorA(0, 0, 0, 0.5f));
			mSafeSpaceTex = gl::Texture2d::create(tbox.render());
		}
		{
			string txt = "For evaluation only. Press a key/button to continue.";
			TextBox tbox = TextBox()
				.alignment(TextBox::CENTER)
				.font(mFont)
				.size(textsize)
				.text(txt);
			tbox.setColor(Color(1.0f, 1.f, 1.f));
			tbox.setBackgroundColor(ColorA(0, 0, 0, 0.5f));
			mEvaluationTex = gl::Texture2d::create(tbox.render());
		}

		if (mShowWorld) {
			mLandtexmix += 0.1 * (mLandtexmixTarget - mLandtexmix);
			if (mLandscapeUpdated) {
				mLandtexmixTarget = !mLandtexmixTarget;
				if (mLandtexmixTarget == 0) {
					mLandTex1->update(world.landscape.ptr(), GL_RGBA,
									 GL_FLOAT,
									 0, mLandTex1->getWidth(),
									 mLandTex1->getHeight(),
									 mLandTex1->getDepth());
				} else {
					mLandTex2->update(world.landscape.ptr(), GL_RGBA,
									 GL_FLOAT,
									 0, mLandTex2->getWidth(),
									 mLandTex2->getHeight(),
									 mLandTex2->getDepth());
				}
				mLandscapeUpdated = false;
			}
		}

		if (mShowParticles) {
			void *gpuMem = mParticleVbo->mapReplace();
			memcpy(gpuMem, world.particle_bases, NUM_PARTICLES * sizeof(particle_base));
			mParticleVbo->unmap();
		}
#ifdef AN_USE_CREATURE
		if (mShowCreatures) {

			// update friends:
			vec3 pos = vec3(world.pos);
			quat cq = glm::inverse(mNavOrient);//  glm::inverse(q);
			glm::mat4 rot = glm::mat4_cast((cq));
			glm::vec3 coffset = glm::vec3(0., -0.1, -0.17);
			if (0) {
				particleCollector& pc = world.collectors[2];
				pc.intensity = 1.f;
				pc.orient = world.collectors[0].orient;
				glm::vec3 cpos = pos + glm::vec3(0., 4., 0.);
				pc.pos = cpos + quat_rotate(pc.orient, coffset);
				creatureInstances[0].instanceMatrix = glm::translate(glm::mat4(), cpos) * glm::mat4_cast(pc.orient);
				creatureInstances[0].instanceMatrixInverse = glm::inverse(creatureInstances[0].instanceMatrix);
			}
			if (0) {
				particleCollector& pc = world.collectors[3];
				pc.intensity = 1.f;
				pc.orient = world.collectors[1].orient;
				glm::vec3 cpos = pos;
				pc.pos = cpos + quat_rotate(pc.orient, coffset);
				creatureInstances[1].instanceMatrix = glm::translate(glm::mat4(), cpos) * glm::mat4_cast(pc.orient);
				creatureInstances[1].instanceMatrixInverse = glm::inverse(creatureInstances[1].instanceMatrix);
			}

			void *gpuMem = mCreatureInstanceDataVbo->mapReplace();
			memcpy(gpuMem, creatureInstances.data(), creatureInstances.size() * sizeof(CreatureInstanceData));
			mCreatureInstanceDataVbo->unmap();
		}
#endif
#ifdef AN_USE_FISH
		{
			void *gpuMem = mFishInstanceDataVbo->mapReplace();
			memcpy(gpuMem, fishInstances.data(), fishInstances.size() * sizeof(FishSegmentInstanceData));
			mFishInstanceDataVbo->unmap();

		}
#endif

		if (mShowOrganisms) {
			// update our instance positions; map our instance data VBO, write new positions, unmap
			mOrganismInstanceCount = 0;
			OrganismInstanceData * instances = (OrganismInstanceData *)mOrganismInstanceDataVbo->mapReplace();
			for (int i = 0; i<NUM_ORGANISMS; i++) {
				const organism& o = world.organisms[i];
				OrganismInstanceData& instance = instances[mOrganismInstanceCount];
				if (o.thing.alive) {
					instance.position = o.thing.pos;
					instance.orient.w = (float)o.orient.w;
					instance.orient.x = (float)o.orient.x;
					instance.orient.y = (float)o.orient.y;
					instance.orient.z = (float)o.orient.z;
					instance.params.x = (float)o.vary;
					instance.params.y = (float)o.flash;
					instance.params.z = (float)(o.thing.nrg - world.organism_decay_threshold);
					mOrganismInstanceCount++;
				}
			}
			mOrganismInstanceDataVbo->unmap();
		}

		if (mShowStalks) {
			// update our instance positions; map our instance data VBO, write new positions, unmap
			mStalkInstanceCount = 0;
			StalkInstanceData * instances = (StalkInstanceData *)mStalkInstanceDataVbo->mapReplace();
			for (int i = 0; i<NUM_STALKS; i++) {
				const stalk& o = world.stalks[i];
				StalkInstanceData& instance = instances[mStalkInstanceCount];
				if (o.thing.alive) {
					const stalk& p = (o.parent != nullptr) ? *o.parent : o;

					instance.position = o.thing.pos;
					instance.orient.w = (float)o.orient.w;
					instance.orient.x = (float)o.orient.x;
					instance.orient.y = (float)o.orient.y;
					instance.orient.z = (float)o.orient.z;
					instance.orient2.w = (float)p.orient.w;
					instance.orient2.x = (float)p.orient.x;
					instance.orient2.y = (float)p.orient.y;
					instance.orient2.z = (float)p.orient.z;
					instance.size = vec2(o.thickness, o.length);
					instance.size2 = vec2(p.thickness, p.length);
					instance.color = o.color;
					instance.color2 = p.color;
					instance.type = (float)o.type;

					mStalkInstanceCount++;
				}
			}
			mStalkInstanceDataVbo->unmap();
		}
	}
	
	void saveImage(ImageSourceRef img, std::string layer = "full") {
		
		std::string filename = "Screenshot";
		
		cinder::fs::path currentPath = ci::getDocumentsDirectory();
		//cinder::fs::path currentPath = ci::getHomeDirectory();
		
		std::ostringstream ss;
		ss << std::setw(10) << std::setfill('0') << mTimestamp;
		ci::fs::path path(currentPath);
		path /= (filename + "_" + ss.str() + "_" + layer + ".png");
		writeImage(path, img);
		std::cout << "File saved: " + path.string() << std::endl;
	}
	
	void saveImage(std::string layer = "full") {
		saveImage(copyWindowSurface(), layer);
	}

	void saveImage(const gl::Texture2dRef & tex, std::string layer = "full") {
		saveImage(tex->createSource(), layer);
	}

	void draw() override {
		Rectf bounds = getWindowBounds();
		glm::ivec2 windowsize = bounds.getSize();
		gl::enableDepthWrite();
		gl::enableDepthRead();
		gl::clear();
#ifdef AN_USE_VR
		if (mVive) {
			{
				hmd::ScopedVive bind{ mVive };
				// binding makes the tracking data valid
				
				// update navigation with this tracking data:
				update_nav();
				mRenderProjectionMatrix = mVive->getCurrentViewProjectionMatrix(vr::Eye_Left);
				
				glm::mat4 hmdmat = mVive->getCurrentViewMatrix();
				glm::quat q = glm::quat_cast(hmdmat);
				// let update_nav know which way HMD is pointing:
				mFaceOrient = q;

				// move to correct position in world:
				quat iq = glm::inverse(mNavOrient);
				mRenderViewMatrix = glm::translate(glm::mat4_cast(iq), glm::vec3(-world.getPos()));

				mVive->renderStereoTargets(std::bind(&EndlessCurrentVRApp::draw_to_vive, this, std::placeholders::_1), mRenderViewMatrix);
			}
		}
#else
		update_nav();
		vec3 p(world.getPos());
		mRenderViewMatrix = glm::lookAt(p, p - quat_uz(mNavOrient), quat_uy(mNavOrient));
		//mRenderViewMatrix = glm::translate(glm::mat4_cast(mNavOrient), glm::vec3(-world.getPos()));
		gl::setViewMatrix(mRenderViewMatrix);
		gl::setProjectionMatrix(glm::perspective(float(M_PI * 0.5), getWindowAspectRatio(), 0.1f, 37.f));
		draw_scene();
#endif
		// now go to 2D drawing:
		gl::disableDepthWrite();
		gl::disableDepthRead();
		// for some reason this is necessary, gl::viewport didn't work 
		glViewport(0, 0, windowsize.x, windowsize.y);
		gl::setMatricesWindow(getWindowSize());
		
#ifdef AN_USE_VR
		if (mVive) {
			// mirror texture
			// change bounds to a square:
			vec2 c = bounds.getCenter();
			float r = std::max(c.x, c.y);
			bounds.set(c.x - r, c.y - r, c.x + r, c.y + r);
			const gl::Texture2dRef& tex = mVive->getEyeTexture(vr::Eye_Left);
			gl::draw(tex, bounds);
		}
#endif
		if (mScreenshot) {
			saveImage();
			draw_scene_to_disk();
			mScreenshot = false;
		}

		if (mShowDebug) {
			gl::drawString("FPS: " + toString(getAverageFps()) 
				+ " \no: " + toString(world.activeorganisms) + " / " + toString(NUM_ORGANISMS) 
				+ " \n@ " + toString(world.pos)
				+ " \nuf " + toString(-quat_uz(mNavOrient))
				+ " \nup " + toString(quat_uy(mNavOrient))
				, vec2(10.0f, 10.0f), Color::white(), mFont);
		}
	}

#ifdef AN_USE_VR
	void draw_to_vive(vr::Hmd_Eye eye)
	{
		draw_scene();
	}
	
	void draw_vive_controllers() {
		glm::mat4 m;
		glm::mat4 controllerCenter = glm::translate(glm::mat4(1.0f), glm::vec3(0., -0.04, 0.06));
		glm::mat4 controllerScale = glm::scale(glm::mat4(1.0f), glm::vec3(0.15, 0.1, 0.25));
		vec3 p = vec3(world.getPos());
		glm::mat4 controllerBase = controllerCenter * controllerScale;


		mat4 nav = glm::mat4_cast(mNavOrient);

		mat4 chap2world = glm::translate(p) * nav;

		mHandShader->uniform("time", (float)getElapsedSeconds());
		gl::ScopedTextureBind texScope1(mNoiseTex, 0);
		
		{
			hmd::HandControllerState state = mVive->getHandController(vr::Eye_Left);
			mHandShader->uniform("hand", 0.f);
			mHandShader->uniform("triggers", glm::vec4(state.trackpad.x, state.trackpad.y, (float)state.trackpadButton, state.trigger));
			if (state.isValid) {
				// move to correct position in world:
				glm::mat4 controller_mat = chap2world * state.pose * controllerBase;
				gl::ScopedModelMatrix push;
				gl::setModelMatrix(controller_mat);
				//gl::drawCoordinateFrame();
				mHandBatch->draw();
			}
		}
		{
			hmd::HandControllerState state = mVive->getHandController(vr::Eye_Right);
			mHandShader->uniform("hand", 1.f);
			mHandShader->uniform("triggers", glm::vec4(state.trackpad.x, state.trackpad.y, (float)state.trackpadButton, state.trigger));
			if (state.isValid) {
				// move to correct position in world:
				glm::mat4 controller_mat = chap2world * state.pose * controllerBase;
				gl::ScopedModelMatrix push;
				gl::setModelMatrix(controller_mat);
				//gl::drawCoordinateFrame();
				mHandBatch->draw();
			}
		}
	}
#endif

	void draw_scene_to_disk() {
		int oversampling = 4;
		int w = 1512 * oversampling;
		int h = 1680 * oversampling;
		gl::Fbo::Format format;
		//format.setSamples( oversampling ); // uncomment this to enable 4x antialiasing
		format.depthTexture();
		//ci::gl::FboRef fbo = gl::Fbo::create(w, h, format);

		ci::gl::FboRef fbo = ci::gl::Fbo::create(w, h, true, true, false);

		gl::enableDepthRead();
		gl::enableDepthWrite();
		{
			gl::ScopedFramebuffer fbScp(fbo);
			gl::ScopedViewport pushvp(fbo->getSize());
			gl::ScopedViewMatrix pushView;
			gl::ScopedProjectionMatrix pushProj;
			glEnable(GL_MULTISAMPLE);

#ifdef AN_USE_VR
			gl::setViewMatrix(mVive->getCurrentViewMatrix(vr::Eye_Left) * mRenderViewMatrix);
			gl::setProjectionMatrix(mVive->getCurrentProjectionMatrix(vr::Eye_Left));
#else
			gl::setViewMatrix(mRenderViewMatrix);
			gl::setProjectionMatrix(glm::perspective(80.f, w/(float)h, 0.1f, 37.f));
#endif
			gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			gl::disableAlphaBlending();
			gl::enableDepthRead();
			gl::enableDepthWrite();
			if (mShowWorld) draw_landscape();
			saveImage(fbo->getColorTexture(), "world");

			gl::disableDepthWrite();
			gl::enableDepthRead();
			gl::enableAlphaBlending();
			glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
			gl::clear(GL_COLOR_BUFFER_BIT);

#ifdef AN_USE_CREATURE
			if (mShowCreatures) draw_creatures();
#endif
			saveImage(fbo->getColorTexture(), "creatures");
			gl::clear(GL_COLOR_BUFFER_BIT);
			draw_particles(oversampling*2); // want bigger particles for screenshots
			saveImage(fbo->getColorTexture(), "particles");
			gl::clear(GL_COLOR_BUFFER_BIT);
			if (mShowStalks) draw_stalks();
			saveImage(fbo->getColorTexture(), "stalks");
			gl::clear(GL_COLOR_BUFFER_BIT);
			if (mShowOrganisms) {
				draw_organisms();
				gl::lineWidth(oversampling);
				draw_organisms_outlines();
			}
			saveImage(fbo->getColorTexture(), "organisms");
			gl::clear(GL_COLOR_BUFFER_BIT);

#ifdef AN_USE_VR
			if (mShowControllers) draw_vive_controllers();
			saveImage(fbo->getColorTexture(), "controllers");
#endif
			gl::enableDepthWrite();
			glDisable(GL_MULTISAMPLE);
		}
	}

	void draw_scene()
	{
		gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		Timer t(true);
		//
		//std::cout << "DRAW SCENE land:" << int(10000 * tland) << "\tparticles: " << int(10000 * tparticles) << "\tskel: " << int(10000 * tskeleton) << "\tstalks: " << int(10000 * tstalks) << "\torganisms: " << int(10000 * torganisms) << std::endl;

		if (mSafeSpace || mEvaluation) {
			//gl::drawCoordinateFrame(0.3f, 0.06f, 0.01f);
			gl::disableDepthRead();
			gl::enableAlphaBlending();
			float h = 1.5f; // height of the center of the image
			gl::ScopedModelMatrix mm0;
			// move to our chaperone center:
			gl::multModelMatrix(glm::translate(world.pos));
			gl::multModelMatrix(glm::translate(h * quat_uy(mNavOrient)));

			// draw a few times around:
			for (float i = 0.f; i < 0.99f; i += 1 / 6.f) {
				float a = M_PI * 2.f * i;
				gl::ScopedModelMatrix mm1;
				gl::multModelMatrix(glm::rotate(a, vec3(0.f, 1.f, 0.f)) * glm::translate(vec3(0.f, 0.f, 1.3f)));

				// this would be the menu panel:
				//const gl::Texture2dRef& tex = mVive->getEyeTexture(vr::Eye_Left);

				float s = 0.25f;
				gl::draw(mSafeSpace ? mSafeSpaceTex : mEvaluationTex, Rectf(s, s, -s, -s));
			}

			// but also draw controllers:
#ifdef AN_USE_VR
			if (mShowControllers) draw_vive_controllers();
#endif
		} else {

			gl::disableAlphaBlending();
			gl::enableDepthRead();
			gl::enableDepthWrite();

			if (mShowWorld) draw_landscape();
			
#ifdef AN_USE_FISH
			draw_fish();
#endif

			gl::disableDepthWrite();
			gl::enableDepthRead();
			gl::enableAlphaBlending();
			//gl::enableAlphaBlending(true);
			//gl::enableAdditiveBlending();
			glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
			//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
#ifdef AN_USE_CREATURE
			if (mShowCreatures) draw_creatures();
#endif		
			draw_particles();
			double tparticles = t.getSeconds(); t.start();

			//gl::enableDepthWrite();

			if (mShowStalks) draw_stalks();
			double tstalks = t.getSeconds(); t.start();
			if (mShowOrganisms) draw_organisms();
			double torganisms = t.getSeconds(); t.start();

#ifdef AN_USE_VR
			if (mShowControllers) draw_vive_controllers();
#endif

		}
		gl::enableDepthWrite();
		gl::enableDepthRead();
	}

	

	void draw_landscape() {
		mLandShader->uniform("now", (float)getElapsedSeconds());
		mLandShader->uniform("landtexmix", mLandtexmix);
		
#ifdef AN_USE_OCULUS
		if (mRift && mDrawOculus) {
			//mLandShader->uniform("pos", vec3(world.pos.x, world.pos.y, world.pos.z) + mRift->getEyeCamera().getEyePoint());
			//glm::quat q = mRift->getEyeCamera().getOrientation();
			//mLandShader->uniform("orient", vec4(q.x, q.y, q.z, q.w));
		}
#endif

		gl::ScopedTextureBind texScope0(mLandTex1, 0);
		gl::ScopedTextureBind texScope1(mLandTex2, 1);
		gl::ScopedTextureBind texScope2(mNoiseTex, 2);
		mLandBatch->draw();
	}


	void draw_particles(float scale = 1.f) {
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
		mParticleShader->uniform("pointSize", scale * 50.f);

		if (mShowParticles) {
			//gl::ScopedTextureBind texScope(mParticleTexture);
			//gl::ScopedTextureBind texScope1(mLandTex, 1);
			mParticleBatch->draw();
		}
		glDisable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
	}

	void draw_organisms() {
		//gl::setWireframeEnabled(true);
		//mOrganismBatch->drawInstanced(mOrganismInstanceCount);
		//gl::setWireframeEnabled(false);
		mOrganismBatch->drawInstanced(mOrganismInstanceCount);
	}

	void draw_organisms_outlines() {
		gl::setWireframeEnabled(true);
		mOrganismBatch->drawInstanced(mOrganismInstanceCount);
		gl::setWireframeEnabled(false);
	}

	void draw_stalks() {
		mStalkBatch->drawInstanced(mStalkInstanceCount);
	}
#ifdef AN_USE_CREATURE
	void draw_creatures() {
		mCreatureShader->uniform("time", (float)getElapsedSeconds());
		gl::ScopedTextureBind texScope1(mNoiseTex, 0);
		mCreatureBatch->drawInstanced((GLsizei)creatureInstances.size());
	}
#endif
#ifdef AN_USE_FISH	
	void draw_fish() {
		//mCreatureShader->uniform("time", (float)getElapsedSeconds());
		//gl::ScopedTextureBind texScope1(mNoiseTex, 0);
		mFishBatch->drawInstanced((GLsizei)fishInstances.size());
	}
#endif

	void mouseDown(MouseEvent event) override {};
	void mouseUp(MouseEvent event) override {};
	void mouseDrag(MouseEvent event) override {};
	void mouseMove(MouseEvent event) override {};
	void mouseWheel(MouseEvent event) override {};

	void keyUp(KeyEvent event) override {
		switch (event.getCode()) {
		case app::KeyEvent::KEY_UP:
		case app::KeyEvent::KEY_w:
		case app::KeyEvent::KEY_DOWN:
		case app::KeyEvent::KEY_s:
			mNavVelocityKeys.z = 0.;
			break;

		case app::KeyEvent::KEY_LEFT:
		case app::KeyEvent::KEY_RIGHT:
			mNavAngularVelocity.y = 0.;
			break;

		case app::KeyEvent::KEY_a:
		case app::KeyEvent::KEY_d:
			mNavVelocityKeys.x = 0.;
			break;

		case app::KeyEvent::KEY_q:
		case app::KeyEvent::KEY_z:
			mNavVelocityKeys.y = 0;
			break;

		default:
			break;
		}
	};

	void keyDown(KeyEvent event) override {
		mSafeSpace = false;
		mEvaluation = false;
		switch (event.getCode()) {
		case KeyEvent::KEY_f:
			goFullScreen();
			break;
		case app::KeyEvent::KEY_UP:
		case app::KeyEvent::KEY_w:
			mNavVelocityKeys.z = -mNavKeyRate.z;
			break;
		case app::KeyEvent::KEY_DOWN:
		case app::KeyEvent::KEY_s:
			mNavVelocityKeys.z = mNavKeyRate.z;
			break;
		case app::KeyEvent::KEY_LEFT:
			mNavAngularVelocity.y = mNavKeyTurnRate.y;
			break;
		case app::KeyEvent::KEY_RIGHT:
			mNavAngularVelocity.y = -mNavKeyTurnRate.y;
			break;

		case app::KeyEvent::KEY_a:
			mNavVelocityKeys.x = -mNavKeyRate.x;
			break;
		case app::KeyEvent::KEY_d:
			mNavVelocityKeys.x = mNavKeyRate.x;
			break;
		case app::KeyEvent::KEY_q:
			mNavVelocityKeys.y = mNavKeyRate.y;
			break;
		case app::KeyEvent::KEY_z:
			mNavVelocityKeys.y = -mNavKeyRate.y;
			break;
#ifdef AN_FINAL
		case KeyEvent::KEY_ESCAPE:
			quit();
			break;

#else 
		case KeyEvent::KEY_ESCAPE:
			// Exit full screen, or quit the application, when the user presses the ESC key.
			if (isFullScreen())
				goFullScreen();
			else
				quit();
			break;
		case KeyEvent::KEY_p:
			mScreenshot = true;
				printf("take screenshot\n");
			break;

		case KeyEvent::KEY_BACKSPACE:
			world.reset();
			break;
		case KeyEvent::KEY_SPACE:
			world.updating = !world.updating;
			break;

		case KeyEvent::KEY_1:
			mShowWorld = !mShowWorld;
			break;
		case KeyEvent::KEY_2:
			mShowParticles = !mShowParticles;
			break;
		case KeyEvent::KEY_3:
			mShowOrganisms = !mShowOrganisms;
			break;
		case KeyEvent::KEY_4:
			mShowStalks = !mShowStalks;
			break;
		case KeyEvent::KEY_5:
			mShowCreatures = !mShowCreatures;
			break;
		case KeyEvent::KEY_6:
			mShowControllers = !mShowControllers;
			break;
		case KeyEvent::KEY_9:
			mShowDebug = !mShowDebug;
			break;
#endif
		}
	}
};


void prepareSettings(App::Settings *settings)
{
#ifdef AN_USE_CONSOLE
{
	int outHandle, errHandle, inHandle;
	FILE *outFile, *errFile, *inFile;
	AllocConsole();

	CONSOLE_SCREEN_BUFFER_INFO coninfo;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &coninfo);
	coninfo.dwSize.Y = 9999;
	SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE), coninfo.dwSize);

	// Redirect the CRT standard input, output, and error handles to the console
	freopen("CONIN$", "r", stdin);
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);

	//Clear the error state for each of the C++ standard stream objects. We need to do this, as
	//attempts to access the standard streams before they refer to a valid target will cause the
	//iostream objects to enter an error state. In versions of Visual Studio after 2005, this seems
	//to always occur during startup regardless of whether anything has been read from or written to
	//the console or not.
	std::wcout.clear();
	std::cout.clear();
	std::wcerr.clear();
	std::cerr.clear();
	std::wcin.clear();
	std::cin.clear();
}
#endif
}

// seems like msaa is not necessary, since we're rendernig to an off-screen texture anyway
//CINDER_APP(EndlessCurrentVRApp, RendererGl(RendererGl::Options().msaa(4)), prepareSettings);
CINDER_APP(EndlessCurrentVRApp, RendererGl(RendererGl::Options()), prepareSettings);
