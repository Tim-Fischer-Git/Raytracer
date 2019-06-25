'''
Created on 28 Mar 2019

@author: Tim Fischer
'''
import PIL.Image
import numpy as np
import threading


class Ray(object):
    
    def __init__(self, origin, direction):
        self.origin = origin #point
        self.direction = normalize(direction) #Vektor
        
    def __repr__(self):
        return "Ray(%s,%s)" %(repr(self.origin), repr(self.direction))
   
    def pointAtParameter(self , t):
        return self.origin + self.direction*t
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm  

class Camera:

    def __init__(self, e, up, c, fieldOfView,width, height):

        self.e = e
        self.up = up
        self.c = c
        self.fieldOfView = fieldOfView

        self.f = normalize(np.subtract(c,e))
        self.s = normalize(np.cross(self.f ,up))
        self.u = np.cross(self.s,self.f)
        
        self.width = width
        self.height = height

        ratio = width / float(height)
        alpha = self.fieldOfView / 2.0
        self.halfHeight = np.tan(alpha)
        self.halfWidth = ratio * self.halfHeight
        self.pixelWidth = self.halfWidth / (width) * 2
        self.pixelHeight = self.halfHeight / (height) * 2

    def calRays(self, x, y):
        xComp = self.s *(x * self.pixelWidth - self.halfWidth)
        yComp = self.u *(y * self.pixelHeight - self.halfHeight)
        return Ray(self.e, self.f + xComp + yComp)
        
class Scene(object):
    
    def __init__(self,camera,image,objLst,backroundColor=[255,255,255],lightSource={"Posion":[6,4,6],"Color":[255,255,255]}):
        '''
        '''
        self.camera = camera
        self.image = image
        self.objLst = objLst
        self.backround = backroundColor
        self.lightsource = lightSource
        
    def renderBeginn(self):
        for x in range(self.camera.width):
                for y in (range(self.camera.height )):
                    g = threading.Thread(target=self.render, args=(x,y))
                    g.start()
    def render(self,x,y):
        #hitdata =self.checkintersecetion(self.camera.calRays(x, y))
        #if hitdata:
            #color =  self.pingPhong(hitdata)
           # color = tuple([int(i) for i  in color ])
        #else:
            #color =self.backround
        color = self.traceRay(self.camera.calRays(x, y)) 
        self.image.putpixel((x,y), tuple(color)) 

    def traceRay(self,ray,level=0):
        hitPointData = self.checkintersecetion(ray,level,2)
        if hitPointData:
            return [int(i) for i in self.shade(level,hitPointData)]
        return self.backround
    
    def shade(self,level , hitPointData ):
        
        directColor = self.pingPhong(hitPointData)
        reflectedRay = self.computeReflectedRay(hitPointData) 
        reflectedColor = self.traceRay(reflectedRay,level+1)
        return directColor + hitPointData["Hitobject"].material.refelction*np.array(reflectedColor)
    
    def computeReflectedRay(self,hitPointData):
      
        n = normalize(hitPointData["Hitobject"].normalAt(hitPointData["Hitpoint"]))
        d = hitPointData["Ray"].direction
        
        return Ray(hitPointData["Hitpoint"],normalize(d)- (2 * np.dot(n,d) * n))
        
    
    def pingPhong(self,hitPointData):
        
        hitobject=hitPointData["Hitobject"]
        hitpoint=hitPointData["Hitpoint"]
        l = normalize(self.lightsource["Posion"]-hitpoint)
        n = normalize(hitobject.normalAt(hitpoint))
        d = normalize(hitPointData["Ray"].direction)
        lr =normalize(l)- (2 * np.dot(n,l) * n)
        
        safe=self.checkintersecetion(Ray(hitpoint,l))
        if safe is None or safe["Hitobject"] is hitobject:
            shadowCoefficient=1
        else:
            shadowCoefficient=0.2
    
            
        ca = hitobject.material.baseColorAt(hitpoint)   
        cin= self.lightsource["Color"]
        ka = hitobject.material.ambientCoefficient
        kd = hitobject.material.diffuseCoefficient
        ks = hitobject.material.specularCoefficient
        
        ambianceShare =np.array(ca)*ka*shadowCoefficient
        diffuseProportion = np.array(cin) * kd*np.dot(l,n)
        speculatorShare  = np.array(cin) *ks*(np.dot(lr,-1*d)**8) 
        # floatresult=((ambianceShare+diffuseProportion+speculatorShare)*(1-hitobject.material.refelction))
        return (ambianceShare+diffuseProportion+speculatorShare)*(1-hitobject.material.refelction)  
                    
    def checkintersecetion(self,ray,reflectionDeep=0, maxReflection = 2):
        hitobject,hitdist = None,None
        maxdist = float(420)
       
        if reflectionDeep == maxReflection:
            return None
        
        for tempOp in self.objLst :
            temHitdist = tempOp.intersectionParameter(ray) 
            if temHitdist :
                if temHitdist < maxdist and temHitdist > 0.000001:

                    hitobject = tempOp
                    maxdist = temHitdist
                    hitdist = temHitdist
        if hitobject:            
            return {"Hitobject":hitobject,"Hitdist":hitdist,"Hitpoint":ray.pointAtParameter(hitdist),"Ray":ray}
        else:
            return None
        
class Material():
    def __init__(self,color,refelction = 0.2,ambianceShare = 0.5,diffuseCoefficient=0.5,specularCoefficient=0.5):
        self.baseColor = color
        self.ambientCoefficient = ambianceShare
        self.diffuseCoefficient = diffuseCoefficient
        self.specularCoefficient = specularCoefficient
        self.refelction = refelction
        
    def baseColorAt(self , p):
        return self.baseColor
    #def GetColor(self,shadowEffect):
        
        #return [i *shadowEffect* self.ambianceShare for i in self.color]
    
class CheckerboardMaterial(object): 
    def __init__(self ):
        self.baseColor = (255, 255, 255) 
        self.otherColor = (0, 0, 0) 
        self.ambientCoefficient = 1.0 
        self.diffuseCoefficient = 0.2 
        self.specularCoefficient = 0.2
        self.refelction = 0 
        self.checkSize = 1
        
    def baseColorAt(self , p):
        v = np.array(p)
        v =v*(1.0 / self.checkSize)
        if (int(abs(v[0]) + 0.5) + int(abs(v[1]) + 0.5) + int(abs(v[2]) + 0.5)) %2:
            return self . otherColor 
        return self . baseColor
                         
class Plane():
    
    def __init__(self, point, normal,material):
        self.point = point # point
        self.normal = normalize(normal)
        self.material = material
    def __repr__(self):
        return 'Plane (%s,%s)' %(repr(self.point), repr(self.normal))
    
    def intersectionParameter(self, ray):
        op = ray.origin - self.point
        a = op.dot(self.normal)
        b = ray.direction.dot(self.normal)
        if b:
            return -a/b
        else:
            return None
        
    def normalAt(self, p):
        return self.normal
    
    def colorAt(self,Ray,sh):
        return self.material.GetColor(sh)
    
class Sphere():
    
    def __init__(self, center, radius,material):
        self.center = center # point
        self.radius = radius # scalar
        self.material= material
        
    def __repr__(self):
        return 'Sphere(%s,%s)' %(repr(self.center), self.radius)
    
    def intersectionParameter(self, ray):
        co = np.subtract(self.center,ray.origin)
        v = np.dot(co,ray.direction)
       
        discriminant = v*v - np.dot(co,co) + self.radius * self.radius
        if discriminant < 0:
            return None
        else:
            return v - np.sqrt(discriminant)
    
    def normalAt(self, p):
        return normalize(p - self.center)
    
    def colorAt(self,ray,sh):
        return self.material.GetColor(sh)

class Triangle():
    def __init__(self , a, b, c,material):
        self.material= material
        self.a = a#point
        self.b = b#point
        self.c = c#point
        self.u = np.subtract(self.b, self.a) # direction vector 
        self .v = np.subtract(self .c ,self .a) # direction vector
    def __repr__(self):
        return "Triangle(%s,%s,%s)" %(repr(self.a), repr(self.b), repr(self.c))
    def intersectionParameter(self , ray): 
        w = np.subtract(ray.origin,self.a)
        dv = np.cross(ray.direction,self.v) 
        dvu = np.dot(dv ,self.u)
        if dvu == 0.0: 
            return None
        wu = np.cross(w,self.u)
        r = np.dot(dv,w) / dvu
        s = np.dot(wu,ray.direction) / dvu
        if 0<=r and r<=1 and 0<=s and s<=1 and r+s<=1:
            return np.dot(wu,self.v) / dvu 
        else :
            return None
    def normalAt( self , p):
        return normalize(np.cross(self.u,self.v))
     
    def colorAt(self,Ray,sh):
        return self.material.GetColor(sh)
    
if __name__ == "__main__":
    objectList = [
      #  Sphere(Point(2.5, 4, -10), 1),
      #Sphere(Point(-2, -2,15), 1),
      Triangle(np.array([1.5, 0, 10]), np.array([-1.5, 0, 10]),
                 np.array([0, 2.4, 10]),Material([255, 255, 0],0.1)),
      Plane(np.array([0, -2, 0]), np.array([0, 1, 0]),CheckerboardMaterial()),#[70,70,70],0)),
      Sphere(np.array([1.2, 0, 9]), 1,Material([255,0,0],0.2)),
      Sphere(np.array([-1.2, 0, 9]),1,Material([0,255,0],0.2)),
      Sphere( np.array([0, 2.2, 9]), 1,Material([0,0,255],0.2))
      #Sphere(Point(0, 7, -10), 10)
    ]
    
    
    image = PIL.Image.new("RGB", (400,400), (255,255,255))
    camera =  Camera(np.array([0, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, 1]), 45,400, 400)
    scene = Scene(camera,image,objectList,[0,0,0])
    scene.renderBeginn()
    image= scene.image
    
    image.show() 
