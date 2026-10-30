"""Build explicit composite CF21B mass asset; visual shapes have no inferred mass."""
from pathlib import Path
import runpy
constants = runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/crazyflie_isaaclab/tasks/crazyflie/constants.py'))
TOTAL_MASS, COM, INERTIA, ROTOR_POS = (constants[k] for k in ('TOTAL_MASS','COM','INERTIA','ROTOR_POS'))
from pxr import Usd, UsdGeom, UsdPhysics, Gf

root = Path(__file__).resolve().parents[1]
evidence = dict(total_mass=TOTAL_MASS, com_native=COM, locked_child_composite_inertia_native=[[INERTIA[0],0,0],[0,INERTIA[1],0],[0,0,INERTIA[2]]], motor_sites={str(i):p for i,p in enumerate(ROTOR_POS)})
path = root/'src/crazyflie_isaaclab/assets/cf21b.usda'
stage = Usd.Stage.CreateNew(str(path))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
body = UsdGeom.Xform.Define(stage,'/CF21B')
stage.SetDefaultPrim(body.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
mass.CreateMassAttr(evidence['total_mass'])
mass.CreateCenterOfMassAttr(Gf.Vec3f(*evidence['com_native']))
j=evidence['locked_child_composite_inertia_native']
mass.CreateDiagonalInertiaAttr(Gf.Vec3f(j[0][0],j[1][1],j[2][2]))
mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0,Gf.Vec3f(0)))
collision=UsdGeom.Sphere.Define(stage,'/CF21B/collision')
collision.CreateRadiusAttr(.086)
collision.AddTranslateOp().Set(Gf.Vec3d(0,0,.02))
collision.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
UsdPhysics.CollisionAPI.Apply(collision.GetPrim())

def cube(name,pos,scale,color,rotation=0):
    obj=UsdGeom.Cube.Define(stage,'/CF21B/'+name)
    obj.CreateSizeAttr(1)
    obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
    obj.AddRotateZOp().Set(rotation)
    obj.AddScaleOp().Set(Gf.Vec3f(*scale))
    obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])

cube('pcb',(0,0,0),(.04,.025,.009),(.05,.35,.12))
cube('battery',(0,0,.012),(.028,.018,.012),(.2,.2,.23))
for i,angle in enumerate((45,-45)):
    cube('arm'+str(i),(0,0,0),(.104,.004,.003),(.12,.12,.12),angle)
for i,p in enumerate(evidence['motor_sites'].values()):
    x,y,_=p
    cyl=UsdGeom.Cylinder.Define(stage,f'/CF21B/motor{i}')
    cyl.CreateRadiusAttr(.0045);cyl.CreateHeightAttr(.012)
    cyl.AddTranslateOp().Set(Gf.Vec3d(x,y,.006))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(.5,.5,.5)])
    color=(.8,.12,.08) if x>0 else (.15,.2,.7)
    cube('prop'+str(i),(x,y,.014),(.035,.005,.001),color,45 if i%2 else -45)
# Red front marker distinguishes native +X.
cube('front',(.023,0,.002),(.006,.01,.004),(.9,.1,.05))
stage.GetRootLayer().Save()
print(path)
