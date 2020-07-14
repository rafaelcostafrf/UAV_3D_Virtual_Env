import panda3d
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight

def world_setup(env, render, mydir):
    env.disableMouse()
    
    # Load the environment model.
    env.scene = env.loader.loadModel(mydir + "/models/city.egg")
    env.scene.reparentTo(env.render)
    env.scene.setScale(1, 1, 1)
    env.scene.setPos(0, 0, 0)
    
    # Load the skybox
    env.skybox = env.loader.loadModel(mydir + "/models/skybox.egg")
    env.skybox.setScale(100,100,100)
    env.skybox.setPos(0,0,-500)
    env.skybox.reparentTo(env.render)

    # Also add an ambient light and set sky color.
    skycol = panda3d.core.VBase3(135 / 255.0, 206 / 255.0, 235 / 255.0)
    env.set_background_color(skycol)
    alight = AmbientLight("sky")
    alight.set_color(panda3d.core.VBase4(skycol * 0.04, 1))
    alight_path = render.attachNewNode(alight)
    render.set_light(alight_path)

    # 4 perpendicular lights (flood light)
    dlight1 = DirectionalLight('directionalLight')
    dlight1.setColor(panda3d.core.Vec4(0.3,0.3,0.3,0.3))
    dlight1NP = render.attachNewNode(dlight1)
    dlight1NP.setHpr(0,0,0)

    dlight2 = DirectionalLight('directionalLight')
    dlight2.setColor(panda3d.core.Vec4(0.3,0.3,0.3,0.3))
    dlight2NP = render.attachNewNode(dlight2)
    dlight2NP.setHpr(-90,0,0)

    dlight3 = DirectionalLight('directionalLight')
    dlight3.setColor(panda3d.core.Vec4(0.3,0.3,0.3,0.3))
    dlight3NP = render.attachNewNode(dlight3)
    dlight3NP.setHpr(-180,0,0)

    dlight4 = DirectionalLight('directionalLight')
    dlight4.setColor(panda3d.core.Vec4(0.3,0.3,0.3,0.3))
    dlight4NP = render.attachNewNode(dlight4)
    dlight4NP.setHpr(-270,0,0)
    render.setLight(dlight1NP)
    render.setLight(dlight2NP)
    render.setLight(dlight3NP)
    render.setLight(dlight4NP)

    # 1 directional light (Sun)
    dlight = DirectionalLight('directionalLight')
    dlight.setColor(panda3d.core.Vec4(1, 1, 1, 1)) # directional light is dim green
    dlight.getLens().setFilmSize(panda3d.core.Vec2(50, 50))
    dlight.getLens().setNearFar(-100, 100)
    dlight.setShadowCaster(True, 4096*2, 4096*2)
    # dlight.show_frustum()
    dlightNP = render.attachNewNode(dlight)
    dlightNP.setHpr(0,-65,0)
    #Turning shader and lights on
    render.setShaderAuto()
    render.setLight(dlightNP)

def quad_setup(env, render, mydir):
    # Load and transform the quadrotor actor.
    env.quad_model = env.loader.loadModel(mydir + '/models/quad.egg')
    env.quad_model.reparentTo(env.render)
    env.prop_1 = env.loader.loadModel(mydir + '/models/prop.egg')
    env.prop_1.setPos(-0.26,0,0)
    env.prop_1.reparentTo(env.quad_model)
    env.prop_2 = env.loader.loadModel(mydir + '/models/prop.egg')
    env.prop_2.setPos(0,0.26,0)
    env.prop_2.reparentTo(env.quad_model)
    env.prop_3 = env.loader.loadModel(mydir + '/models/prop.egg')
    env.prop_3.setPos(0.26,0,0)
    env.prop_3.reparentTo(env.quad_model)
    env.prop_4 = env.loader.loadModel(mydir + '/models/prop.egg')
    env.prop_4.setPos(0,-0.26,0)
    env.prop_4.reparentTo(env.quad_model)
    
    env.prop_models = (env.prop_1, env.prop_2, env.prop_3, env.prop_4)
    
    #env cam
    env.cam_neutral_pos = panda3d.core.LPoint3f(5, 5, 7)
    env.cam.node().getLens().setFilmSize(36, 24)
    env.cam.node().getLens().setFocalLength(45)
    env.cam.setPos(env.cam_neutral_pos)
    env.cam.reparentTo(env.render)
    env.cam.lookAt(env.quad_model)
    
    env.checker = env.loader.loadModel(mydir + '/models/checkerboard.egg')
    env.checker.reparentTo(env.render)
    env.checker_scale = 0.5
    env.checker_sqr_size = 0.2046
    env.checker.setScale(env.checker_scale, env.checker_scale, 1)
    env.checker.setPos(3*env.checker_scale*env.checker_sqr_size+0.06, 2.5*env.checker_scale*env.checker_sqr_size+0.06, 0.001)
    