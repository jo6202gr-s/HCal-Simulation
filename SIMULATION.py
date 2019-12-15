# -*- coding: utf-8 -*-

from math import *
import random
import numpy as np
from decimal import *
import traceback
import array as arr
import matplotlib.pyplot as plt
import scipy.stats
from collections import Counter

"""
Created on Mon Oct 14 12:33:33 2019

@author: jgreaves

Additions/changeLog:
    Incrementing Particle position calculated inside photonImpact method. DONE

    Edge cases - when Cos(newDir) is greater than distance to edge of bar. DONE
    Add logic to check fibre volume interaction DONE
    Streamline: consolidate Photon Impact and Photon Scatter methods[DONE].
    Test Scattering  [DONE] ( and TEST Distance Travelled) (how to keep track of time?)
    Rounding positions and incremnets to 6 DP to prevent division by zero. 
        - This should pertain to all places where coordinates are adjusted.
    Rounding directionTheta to 2DP when comparing against pi, to prevent overflow during scattering.
    Some post processing required during simulation to remove potentially overflowing variables - distance travelled coming out at ~10^32metres
        - This is achieved by setting erroneous values to an arrival time of -1. Values are binned between 0 and a maximum, so -1's are ignored.
NEXT TASKS:
    Introduce Boundary characteristics (some potential additional parameters)
        -Assuming diffuse reflection at bar edges. (with 100% prob) 
        -Beginning assumption ZERO prob to be absorbed in plastic(at edges)
            - perhaps change to disappear photon after certain number of scatters to simulate loss.
        -Currently transmitting photons born inside fibre volume. Perhaps stop this, since none will be born there.
    Introduce Fibre volume DONE
    Extend to 3 Dimensions. DONE
    
IDEAS
-Make photon object an instance of photon called photon2D, that extends and inherits from the Bar class. Use 'Super' to call photonImpact method. DONE
-Set limits for photon position to break loop. [DONE]
-When expanding to 3D, bar method of photonImpact (now just scatterPhoton() ) needs to include y logic. (Spherical co ords? New direction could be in spherical co ords!)
-Particle has register of travelled distance, which is added to each scatter.Perhaps count scatters in another variable.Loop until "transmitted" is TRUE. 
-Is fibre volume in path (or initially 100 scatters for example), then set transmit to true.
-Nest all this in an "event" loop that does n "transmits", where n is number of photons from a scintillation event.
-To find fibre:only depends on y and z. If both coordinates will be between certain limits at the same time, then fibre has been met.
    Idea: pick y or z, and movePhoton until Edge of fibre met.
        In 2D: is either fibre limit between startZ and endZ. 
        in 3D, only two potential points on the fibre circumference will be crossed for the first (z) coordinate. 
        If true, check y coordinate when z = either point (in priority order based on trajectory)
-Bucket photon arrivals to allow superposition. Produce an event plot with photon counts stacking on eachother.  DONE
-Introduce 2 ends 
-Photon reflection at fibre end
-Maybe create photons along a vector of energy deposit by the cosmic, since they won't all be created in one spot.
-Consider Optical properties at fibre boundary 
- add modelling of the PMT response (Poisson), see paper  "PMT response model"

Parameter:Remove reflections at the end of the bar. 
            Inhibit reflections using a probability at the edge of the bar, and the edge of the fibre. ADDING AN AIR GAP! This will reduce transmitted photons alot.
            Add Attenuation of fibre
    
    Photon statistics: Model exponential (with fast and slow components) of photon re emission process during scintillation.
        Maybe this will make it appear more like a (positively skewed) poission distribution??
    
    Create photon collection/measurement
    Convert event energy into amount of photons, and set them off in the Bar.
    

    
    
"""





###############################################################################
"""
INSTRUCTIONS:

Begin at section labelled "ADJUSTABLE VARIABLES", towards the end, after the simulation code.
There is a line to hardcode the number of scintillation photons 'numPhotons' 
    (if it gets calculated from the energy deposited by the event, we have ~10^6 photons)

Every photon is produced and scatters around until it is transmitted, or lost.

I have introduced the following parameters:
    Probability to be lost at the bar boundary
    Probability to reflect instead of transmit at the fibre
    Removing all photons that hit the ends of the bar, since there is no reflective coating there.
    
I could introduce Geometric logic to determine, for example, scattering off the fibre, but it would likely be a small correction with so many photons,
and aprroximable by a probability. 
    

"""

###############################################################################
# Initialize Bar Class #
###############################################################################
class Bar:
    dictInc = {'xInc': 0, 'yInc': 0, 'zInc': 0, 'wall': '0'} #Dictionary to hold increments of coordinates
    distanceTravelled = 0

    def __init__(self, length, width, height, fibreRadius):
        self.length = length
        self.width = width
        self.height = height
        self.fibreRadius = fibreRadius


    def updateIncrements(self, xInc, yInc, zInc, surface):
        self.distanceTravelled = self.distanceTravelled + sqrt(xInc**2 + yInc**2 + zInc**2)
        self.dictInc.update({'xInc': xInc, 'yInc': yInc, 'zInc': zInc, 'wall': surface})




    def fibreInYPath(self, directionPhi, xPos, yPos, zPos):

        if yPos > self.fibreYMin:                            # Y position is behind fibre
            if directionPhi <= pi or directionPhi == 2*pi:
                fibreInYPath = False
            else:
                fibreInYPath = True
        elif yPos < self.fibreYMax:                            # Y position is in front of fibre
            if directionPhi >= pi or directionPhi == 2*pi:
                fibreInYPath = False
            else:
                fibreInYPath = True
        

        else:
            fibreInYPath = False

        return fibreInYPath
    
    
    
    def fibreInZPath(self, directionTheta, xPos, yPos, zPos):

        if zPos > self.fibreZMin:                              # Z position is above fibre
            if directionTheta <= pi/2:
                fibreInZPath = False
            else:
                fibreInZPath = True
        elif zPos < self.fibreZMax:                            # Z position is in Below fibre
            if directionTheta >= pi/2:
                fibreInZPath = False
            else:
                fibreInZPath = True
        else:
            fibreInZPath = False

        return fibreInZPath
    
    def fibreInZ_YPath(self, yPos, zPos, yInc, zInc):
        # creating a 2D projection onto the y-z axis using known points, and finding intersection with fibre circle
        if(yInc == 0):
            grad = 0
        else:
            grad = ((zPos+zInc) - zPos) / ((yPos+yInc) - yPos)
            
        intercept = zPos - grad * yPos
        
        
#        print("Quadratic function : (a * y^2) + b*y + c")
        a = 1 + grad**2
        b = (2 * grad * intercept) - self.width - (self.height * grad)
        c = intercept**2 + (self.height**2 + self.width**2)/4 - (self.height * intercept) - self.fibreRadius**2
        
        det = b**2 - 4*a*c
        
        if det > 0:
            num_roots = 2
            y1 = (((-b) + sqrt(det))/(2*a))     
            y2 = (((-b) - sqrt(det))/(2*a))
#            print("There are 2 roots: %f and %f" % (y1, y2))
            if(abs(y1 - yPos) < abs(y2 - yPos)):                      # Set new Y position to closest fibre interaction point
                y = y1
            else:
                y = y2
            z = grad * y + intercept
#            print("yPos: %f, zPos: %f" % (y, z))
            fibreInZ_YPath = True
        elif det == 0:
            num_roots = 1
            y = (-b) / 2*a
#            print("There is one root: ", y)
            z = grad * y + intercept
            fibreInZ_YPath = True
        else:
            num_roots = 0
#            print("No roots, discriminant < 0.")
            y = yPos + yInc
            z = zPos + zInc
            fibreInZ_YPath = False
            

        return fibreInZ_YPath, round(y-yPos, 6), round(z-zPos , 6)   # these incerements will return 0 if photon not to be transmitted

    def losePhoton(self):
        self.distanceTravelled = 0
        self.dictInc.update({'xInc': 0, 'yInc': 0, 'zInc': 0, 'wall': 'Lost'})


# Calculate coordinates of where the photon meets the fibre
    def transmitPhoton(self, directionPhi, directionTheta, xPos, xInc, yInc, zInc):
        global blnTransmitted

        # Quadrant direction 1
        if directionPhi < pi/2:
            xDistToFibre = yInc/tan(directionPhi)
            xInc = xDistToFibre
        # Quadrant direction 2
        elif directionPhi > pi/2 and directionPhi < pi:
            xDistToFibre = yInc/tan(pi-directionPhi)
            xInc = xDistToFibre
        # Quadrant direction 3
        elif directionPhi > pi and directionPhi < 3*pi/2:
            xDistToFibre = tan((3*pi/2)-directionPhi)*yInc
            xInc = xDistToFibre  
        # Quadrant direction 4
        elif directionPhi > 3*pi/2 and directionPhi < 2*pi:
            xDistToFibre = tan(directionPhi - (3*pi/2))*yInc
            xInc = xDistToFibre

        # UP          
        elif directionPhi == pi/2:
            xInc = 0
        # LEFT                         
        elif directionPhi == pi:
            if directionTheta < pi/2:
                xInc = -1 * tan(directionTheta) * zInc
            else:
                xInc = -1 * tan(pi - directionTheta) * zInc
        # DOWN        
        elif directionPhi == 3*pi/2:
            xInc = 0
        # RIGHT         
        elif directionPhi == 2*pi:
            if directionTheta < pi/2:
                xInc = tan(directionTheta) * zInc
            else:
                xInc = tan(pi - directionTheta) * zInc
   
        # It is necessary to remove extremely small precision numbers
        xInc = round(xInc,6)
        yInc = round(yInc,6)
        zInc = round(zInc,6)
     
        surface = 'Fibre'
        self.distanceTravelled = self.distanceTravelled + xPos  # add length of fibre to distance.
        self.updateIncrements(xInc, yInc, zInc, surface)
        blnTransmitted = True
        
    
    
     
    def scatterPhoton(self, directionPhi, directionTheta, xPos, yPos, zPos):
        global blnAtFibreEdge
        global blnPhotonLost
        global blnTransmitted  # ensure to set the global boolean variable in this method.
        global exception
        # Corner hits will never happen due to tiny differences in calculated values. They will never be exactly equal. 
        # As a result, I have not coded for corners; they will be accounted for one way or another.
        
        xInc = 0
        yInc = 0
        zInc = 0
        
        
        if xPos>self.length or yPos>self.width or zPos>self.height:
            return "ERROR - wrong dimensions. Particle is outside of Bar."
        
        # These are 'self' because they belong to the Photon class. They can now be used in called functions.
        self.distToTop=self.height - zPos
        self.distToRight=self.length - xPos
        self.distToBack = self.width - yPos
        self.distToLeft = xPos
        self.distToBottom = zPos
        self.distToFront = yPos
        self.fibreZMin = (self.height/2)-self.fibreRadius
        self.fibreZMax = (self.height/2)+self.fibreRadius
        self.fibreYMin = (self.width/2)-self.fibreRadius
        self.fibreYMax = (self.width/2)+self.fibreRadius
        
        
        if (zPos > self.fibreZMin and zPos < self.fibreZMax) and (yPos > self.fibreYMin and yPos < self.fibreYMax):
            blnTransmitted = True
            surface = 'Fibre'
            self.updateIncrements(0, 0, 0, surface)
        # This code should never be hit - photon should never be inside the fibre volume without being transmitted.
        
###############################################################################
# Quadrant direction 1

        if directionPhi < pi/2:  
            # Octant direction 1                                # Theta values 0, pi/2, and pi are dealt with separately
            if directionTheta < pi/2 and directionTheta >= 0:   # UPWARD
                if(directionTheta == 0):
                    vecR = self.distToTop
                else:
                    vecR = self.distToRight/(sin(directionTheta)*cos(directionPhi)) # determine if Right or Back is in photon path
                                                                                    # values for z and y compared to x being the distance to the right wall.
                x = vecR * sin(directionTheta) * cos(directionPhi)
                y = vecR * sin(directionTheta) * sin(directionPhi)
                z = vecR * cos(directionTheta)
                if(vecR == 0):
                    yOvershoot = 0
                    zOvershoot = 0
                if(abs(y) > self.distToBack):               # then it will hit the top or back wall
                    yOvershoot = self.distToBack/abs(y)     # how much did estimate overshoot back wall by
                    z = z * yOvershoot                      # Also reduce z and x by this much
                    x = x * yOvershoot 
                    if(abs(z) > self.distToTop):
                        zOvershoot = self.distToTop/abs(z)  # same logic for overshooting top wall
                        surface = "TOP"
                        zInc = self.distToTop
                        xInc = x * zOvershoot
                        yInc = self.distToBack * zOvershoot # equivalent to y * yOvershoot * zOvershoot
                    else:
                        surface = "BACK"                    # here there is no z overshoot
                        zInc = z                            # z has still been reduced since we are scattering off the back wall. 
                        xInc = x
                        yInc = self.distToBack
                else:                                       # then it will hit the top or right wall
                    if(abs(z) > self.distToTop):
                        zOvershoot = self.distToTop/abs(z)
                        surface = "TOP"
                        zInc = self.distToTop
                        xInc = x * zOvershoot
                        yInc = y * zOvershoot
                    else:                                   # No overshoots
                        surface = "RIGHT"
                        xInc = x
                        yInc = y
                        zInc = z
            
                        
            # Octant direction 2
            elif directionTheta >= pi/2 and round(directionTheta,2) <= pi:   # DOWNWARD
                if(directionTheta == pi):
                    vecR = self.distToBottom
                else:
                    vecR = self.distToRight/abs((sin(directionTheta)*cos(directionPhi)))
                x = vecR * sin(directionTheta) * cos(directionPhi)
                y = vecR * sin(directionTheta) * sin(directionPhi)
                z = vecR * cos(directionTheta)
                if(vecR == 0):
                    yOvershoot = 0
                    zOvershoot = 0
                if(abs(y) > self.distToBack):
                    yOvershoot = self.distToBack/abs(y)
                    z = z * yOvershoot
                    x = x * yOvershoot
                    if(abs(z) > self.distToBottom):
                        zOvershoot = self.distToBottom/abs(z)
                        surface = "BOTTOM"
                        zInc = -1 * self.distToBottom
                        xInc = x * zOvershoot
                        yInc = self.distToBack * zOvershoot
                    else:
                        surface = "BACK"
                        zInc = z
                        xInc = x
                        yInc = self.distToBack
                else:
                    if(abs(z) > self.distToBottom):
                        zOvershoot = self.distToBottom/abs(z)
                        surface = "BOTTOM"
                        zInc = -1 * self.distToBottom
                        xInc = x * zOvershoot
                        yInc = y * zOvershoot
                    else:
                        surface = "RIGHT"
                        xInc = x
                        yInc = y
                        zInc = z
###############################################################################        
###############################################################################
# Quadrant direction 2           

        elif directionPhi >= pi/2 and directionPhi < pi:            
            # Octant direction 3
            if directionTheta < pi/2 and directionTheta >= 0:                            # determine if Left or Back is in photon path
                if(directionTheta == 0):
                    vecR = self.distToTop
                else:
                    vecR = abs(self.distToLeft/(sin(directionTheta)*cos(directionPhi)))  # values for z and y compared to x being the distance to the left wall.
                x = vecR * sin(directionTheta) * cos(directionPhi)
                y = vecR * sin(directionTheta) * sin(directionPhi)
                z = vecR * cos(directionTheta)
                if(vecR == 0):
                    yOvershoot = 0
                    zOvershoot = 0
                if(abs(y) > self.distToBack):
                    yOvershoot = self.distToBack/abs(y)
                    z = z * yOvershoot
                    x = x * yOvershoot 
                    if(abs(z) > self.distToTop):
                        zOvershoot = self.distToTop/abs(z)
                        surface = "TOP"
                        zInc = self.distToTop
                        xInc = x * zOvershoot
                        yInc = self.distToBack * zOvershoot
                    else:
                        surface = "BACK"
                        zInc = z
                        xInc = x
                        yInc = self.distToBack
                else:
                    if(abs(z) > self.distToTop):
                        zOvershoot = self.distToTop/abs(z)
                        surface = "TOP"
                        zInc = self.distToTop
                        xInc = x * zOvershoot
                        yInc = y * zOvershoot
                    else:
                        surface = "LEFT"
                        xInc = x
                        yInc = y
                        zInc = z

                        
            # Octant direction 4
            elif directionTheta >= pi/2 and round(directionTheta,2) <= pi:
                if(directionTheta == pi):
                    vecR = self.distToBottom
                else:
                    vecR = abs(self.distToLeft/abs((sin(directionTheta)*cos(directionPhi))))
                x = vecR * sin(directionTheta) * cos(directionPhi)
                y = vecR * sin(directionTheta) * sin(directionPhi)
                z = vecR * cos(directionTheta)
                if(vecR == 0):
                    yOvershoot = 0
                    zOvershoot = 0
                if(abs(y) > self.distToBack):
                    yOvershoot = self.distToBack/abs(y)
                    z = z * yOvershoot
                    x = x * yOvershoot
                    if(abs(z) > self.distToBottom):
                        zOvershoot = self.distToBottom/abs(z)
                        surface = "BOTTOM"
                        zInc = -1 * self.distToBottom
                        xInc = x * zOvershoot
                        yInc = self.distToBack * zOvershoot
                    else:
                        surface = "BACK"
                        zInc = z
                        xInc = x
                        yInc = self.distToBack
                else:
                    if(abs(z) > self.distToBottom):
                        zOvershoot = self.distToBottom/abs(z)
                        surface = "BOTTOM"
                        zInc = -1 * self.distToBottom
                        xInc = x * zOvershoot
                        yInc = y * zOvershoot
                    else:
                        surface = "LEFT"
                        xInc = x
                        yInc = y
                        zInc = z
                    
###############################################################################
###############################################################################            
# Quadrant direction 3

        elif directionPhi >= pi and directionPhi < 3*pi/2:
            # Octant direction 5
            if directionTheta < pi/2 and directionTheta >= 0:
                if(directionTheta == 0):
                    vecR = self.distToTop
                else:
                    vecR = abs(self.distToLeft/(sin(directionTheta)*cos(directionPhi)))
                x = vecR * sin(directionTheta) * cos(directionPhi)
                y = vecR * sin(directionTheta) * sin(directionPhi)
                z = vecR * cos(directionTheta)
                if(vecR == 0):
                    yOvershoot = 0
                    zOvershoot = 0
                if(abs(y) > self.distToFront):
                    yOvershoot = self.distToFront/abs(y)
                    z = z * yOvershoot
                    x = x * yOvershoot 
                    if(abs(z) > self.distToTop):
                        zOvershoot = self.distToTop/abs(z)
                        surface = "TOP"
                        zInc = self.distToTop
                        xInc = x * zOvershoot
                        yInc = -1 * self.distToFront * zOvershoot
                    else:
                        surface = "FRONT"
                        zInc = z
                        xInc = x
                        yInc = -1 * self.distToFront
                else:
                    if(abs(z) > self.distToTop):
                        zOvershoot = self.distToTop/abs(z)
                        surface = "TOP"
                        zInc = self.distToTop
                        xInc = x * zOvershoot
                        yInc = y * zOvershoot
                    else:
                        surface = "LEFT"
                        xInc = x
                        yInc = y
                        zInc = z

                        
            # Octant direction 6
            elif directionTheta >= pi/2 and round(directionTheta,2) <= pi:
                if(directionTheta == pi):
                    vecR = self.distToBottom
                else:
                    vecR = abs(self.distToLeft/abs((sin(directionTheta)*cos(directionPhi))))
                x = vecR * sin(directionTheta) * cos(directionPhi)
                y = vecR * sin(directionTheta) * sin(directionPhi)
                z = vecR * cos(directionTheta)
                if(vecR == 0):
                    yOvershoot = 0
                    zOvershoot = 0
                if(abs(y) > self.distToFront):
                    yOvershoot = self.distToFront/abs(y)
                    z = z * yOvershoot
                    x = x * yOvershoot
                    if(abs(z) > self.distToBottom):
                        zOvershoot = self.distToBottom/abs(z)
                        surface = "BOTTOM"
                        zInc = -1 * self.distToBottom
                        xInc = x * zOvershoot
                        yInc = -1 * self.distToFront * zOvershoot
                    else:
                        surface = "FRONT"
                        zInc = z
                        xInc = x
                        yInc = -1 * self.distToFront
                else:
                    if(abs(z) > self.distToBottom):
                        zOvershoot = self.distToBottom/abs(z)
                        surface = "BOTTOM"
                        zInc = -1 * self.distToBottom
                        xInc = x * zOvershoot
                        yInc = y * zOvershoot
                    else:
                        surface = "LEFT"
                        xInc = x
                        yInc = y
                        zInc = z
                    
###############################################################################
###############################################################################            
# Quadrant direction 4

        elif directionPhi >= 3*pi/2 and directionPhi <= 2*pi:
            # Octant direction 7
            if directionTheta < pi/2 and directionTheta >= 0:
                if(directionTheta == 0):
                    vecR = self.distToTop
                else:
                    vecR = abs(self.distToRight/(sin(directionTheta)*cos(directionPhi)))
                x = vecR * sin(directionTheta) * cos(directionPhi)
                y = vecR * sin(directionTheta) * sin(directionPhi)
                z = vecR * cos(directionTheta)
                if(vecR == 0):
                    yOvershoot = 0
                    zOvershoot = 0
                if(abs(y) > self.distToFront):
                    yOvershoot = self.distToFront/abs(y)
                    z = z * yOvershoot
                    x = x * yOvershoot 
                    if(abs(z) > self.distToTop):
                        zOvershoot = self.distToTop/abs(z)
                        surface = "TOP"
                        zInc = self.distToTop
                        xInc = x * zOvershoot
                        yInc = -1 * self.distToFront * zOvershoot
                    else:
                        surface = "FRONT"
                        zInc = z
                        xInc = x
                        yInc = -1 * self.distToFront
                else:
                    if(abs(z) > self.distToTop):
                        zOvershoot = self.distToTop/abs(z)
                        surface = "TOP"
                        zInc = self.distToTop
                        xInc = x * zOvershoot
                        yInc = y * zOvershoot
                    else:
                        surface = "RIGHT"
                        xInc = x
                        yInc = y
                        zInc = z

                        
            # Octant direction 8
            elif directionTheta >= pi/2 and round(directionTheta,2) <= pi:
                if(directionTheta == pi):
                    vecR = self.distToBottom
                else:
                    vecR = abs(self.distToRight/abs((sin(directionTheta)*cos(directionPhi))))
                x = vecR * sin(directionTheta) * cos(directionPhi)
                y = vecR * sin(directionTheta) * sin(directionPhi)
                z = vecR * cos(directionTheta)
                if(vecR == 0):
                    yOvershoot = 0
                    zOvershoot = 0
                if(abs(y) > self.distToFront):
                    yOvershoot = self.distToFront/abs(y)
                    z = z * yOvershoot
                    x = x * yOvershoot
                    if(abs(z) > self.distToBottom):
                        zOvershoot = self.distToBottom/abs(z)
                        surface = "BOTTOM"
                        zInc = -1 * self.distToBottom
                        xInc = x * zOvershoot
                        yInc = -1 * self.distToFront * zOvershoot
                    else:
                        surface = "FRONT"
                        zInc = z
                        xInc = x
                        yInc = -1 * self.distToFront
                else:
                    if(abs(z) > self.distToBottom):
                        zOvershoot = self.distToBottom/abs(z)
                        surface = "BOTTOM"
                        zInc = -1 * self.distToBottom
                        xInc = x * zOvershoot
                        yInc = y * zOvershoot
                    else:
                        surface = "RIGHT"
                        xInc = x
                        yInc = y
                        zInc = z
                    
###############################################################################
###############################################################################            
# Other directions

        # FRONT          
        elif directionPhi == 3*pi/2 and directionTheta == pi/2:
            surface = "FRONT" 
            xInc = 0
            yInc = -1 * self.distToFront
            zInc = 0
        # BACK          
        elif directionPhi == pi/2 and directionTheta == pi/2:
            surface = "BACK" 
            xInc = 0
            yInc = self.distToBack
            zInc = 0
        # LEFT          
        elif directionPhi == pi and directionTheta == pi/2:
            surface = "LEFT" 
            xInc = -1 * self.distToLeft
            yInc = 0
            zInc = 0
        # RIGHT          
        elif directionPhi == 3*pi/2 and directionTheta == pi/2:
            surface = "RIGHT" 
            xInc = self.distToRight
            yInc = 0
            zInc = 0
        # TOP          
        elif directionTheta == 0:
            surface = "TOP" 
            xInc = 0
            yInc = 0
            zInc = self.distToTop
        # BOTTOM          
        elif directionTheta == pi:
            surface = "BOTTOM" 
            xInc = 0
            yInc = 0
            zInc = -1 * self.distToBottom
###############################################################################            
        fibreInYPath = self.fibreInYPath(directionPhi, xPos, yPos, zPos)
        fibreInZPath = self.fibreInZPath(directionTheta, xPos, yPos, zPos)
        
        # It is necessary to remove extremely small precision numbers to prevent division by 'zero' later.
        xInc = round(xInc,6)
        yInc = round(yInc,6)
        zInc = round(zInc,6)
        xPos = round(xPos,6)
        yPos = round(yPos,6)
        zPos = round(zPos,6)

        if(surface == 'LEFT' or surface == 'RIGHT') and (decision(probReflectBarEnd) == False):
            blnPhotonLost = True
        elif(surface == 'TOP' or surface == 'BOTTOM' or surface == 'FRONT' or surface == 'BACK') and (decision(probReflectBarEdge) == False):
            blnPhotonLost = True

        if(fibreInYPath == True and fibreInZPath == True):
            fibreInZ_YPath, yInc, zInc = self.fibreInZ_YPath(yPos, zPos, yInc, zInc)
        else:
            fibreInZ_YPath = False
        
        if(fibreInZ_YPath == True):
            blnAtFibreEdge = True

        if blnAtFibreEdge == False:
            if blnPhotonLost == False:
                self.updateIncrements(xInc, yInc, zInc, surface)
                self.scatterSurface = surface
            else:
                self.losePhoton()
        else:
            if(decision(probTransmitFibreEdge)):
                blnTransmitted = True
                self.transmitPhoton(directionPhi, directionTheta, xPos, xInc, yInc, zInc)     # we only need to determine xInc, from yInc
            else:
                blnTransmitted = False
                self.updateIncrements(xInc, yInc, zInc, surface)
                self.scatterSurface = surface
                


        self.scatterSurface = self.dictInc['wall']
        
        
        return self.dictInc

###############################################################################
# END Bar Class #      
###############################################################################
    


###############################################################################
# Initialize Photon Class #
###############################################################################
#Initialize Photon class, for photon in the bar, with energy in GeV and position as a dictionary, (distance travelled?).
class Photon(Bar):
    numScatters = 0
    scatterSurface = 0
    arrivalTime = 0  # after event
    
    def __init__(self, dictPos, energy, barX, barY, barZ, fibreRadius):
        self.dictPos = dictPos  # PositionDictionary containing xPos,yPos,zPos
        self.energy= energy
        self.directionPhi = 0
        self.directionTheta = 0
        super().__init__(barX, barY, barZ, fibreRadius)  # Initialize parent bar parameters
    
    
    def movePhoton(self):
        global exception
        #Determine direction of photon propogation
        if(self.numScatters == 0):
            self.directionPhi, self.directionTheta = new_free_direction()
        else:
            self.directionPhi, self.directionTheta = new_scatter_direction(self.scatterSurface)


        #Proceed to next interaction, i.e. with Wall of Bar (or boundary of fibre). 
        try:
            super().scatterPhoton(self.directionPhi,self.directionTheta, self.dictPos['xPos'],self.dictPos['yPos'],self.dictPos['zPos'])
        except Exception as e:
            traceback.print_exc()
            print("Exception: " + str(e) + ". Exiting simulation here. See traceback immediately following. Current photon coordinates: " + str((self.dictPos['xPos'],self.dictPos['yPos'],self.dictPos['zPos'])))
            exception = True
        
        
        xInc = super().dictInc['xInc']
        yInc = super().dictInc['yInc']
        zInc = super().dictInc['zInc']
        
        
        #Adjust coordinates
        newX= (self.dictPos['xPos'])+(xInc)
        newY= (self.dictPos['yPos'])+(yInc)
        newZ= (self.dictPos['zPos'])+(zInc)
        
        # It is necessary to remove extremely small precision numbers
        # abs() to prevent any erroneous small negative values
        newX = round(abs(newX),6)
        newY = round(abs(newY),6)
        newZ = round(abs(newZ),6)

        
        #update position dictionary
        self.dictPos.update({'xPos':newX})
        self.dictPos.update({'yPos':newY})
        self.dictPos.update({'zPos':newZ})
        
        if(blnPhotonLost == False):
            self.numScatters += 1
        else:
            self.numScatters = 0
            
    # Return photon position
    def photonPosition(self):
        strPos = "Photon's current position: x=" + str(self.dictPos['xPos']) + ", y=" + str(self.dictPos['yPos']) + ", z=" + str(self.dictPos['zPos'])
        return strPos
    
    # Return Increments to coordinates
    def printIncrements(self):
        strIncs = "Coordinate Increments: "  + str(super().dictInc) + "\n"
        return (strIncs)
###############################################################################
# END Photon Class #
###############################################################################



###############################################################################    
# Global Methods #
###############################################################################   

def decision(probability):
    rand = random.random()
    return rand < probability


def new_free_direction():
    randPhi = round(radians(random.randint(0,360)),6)
    if randPhi == 0:
        randPhi = 2*pi
        
    randTheta = round(radians(random.randint(0,180)),6)
       
#    randPhi = 0.10241638*pi
#    randTheta = 0.32043*pi

    return randPhi, randTheta

def new_scatter_direction(wall):
        
    if wall == "RIGHT":
        randPhi = radians(random.randint(90,270))
        randTheta = radians(random.randint(0,180))
    elif wall == "LEFT":
        listAngles = [random.randint(0,90),random.randint(270,360)]
        randPhi = radians(random.choice(listAngles))
        randTheta = radians(random.randint(0,180))
    elif wall == "BACK":
        randPhi = radians(random.randint(180,360))
        randTheta = radians(random.randint(0,180))
    elif wall == "FRONT":
        randPhi = radians(random.randint(0,180))
        randTheta = radians(random.randint(0,180))
    elif wall == "TOP":
        randPhi = radians(random.randint(0,360))
        randTheta = radians(random.randint(90,180))
    elif wall == "BOTTOM":
        randPhi = radians(random.randint(0,360))
        randTheta = radians(random.randint(0,90))
    else:
        randPhi = 0
        
    if randPhi == 0:
        randPhi = 2*pi
    
    return round(randPhi,6), round(randTheta,6)

###############################################################################
# END Global Methods #
###############################################################################




###############################################################################
# SIMULATION # and definition of global variables
###############################################################################
blnTransmitted = False
exception = False
blnAtFibreEdge = False
blnPhotonLost = False

#### ADJUSTABLE VARIABLES ####    
speedOfLight = 186.4                  # = 300  # mm per ns
phEnergy = 2.61                     # Scintillation Photon Energy (eV)
evEnergy = 2*10**6                  # Energy deposited by Event (eV)
numPhotons = (evEnergy/phEnergy)/100  # Number of scinitillation photns produced by event
barLength = 1000                    # mm
barWidth = 50
barHeight = 20
fibreRadius= 1.4                    # mm
peVoltage = -20*10**-3               # Average voltage of single photo electron [V]
quantumEfficiency = 0.235            # QE of the photocathode in the PMT (source Photonis datasheet)
window = 40*10**-9                  # time window in seconds, 100ns = length of the window on the oscilloscope.
binWidth = 0.001                       # ns

probReflectBarEnd = 0         # Lost photons if they hit left or right ends of the bar
probReflectBarEdge = 0.95
probTransmitFibreEdge = 0.25  # simulate Air gap
###############################

# Set numPhotons for testing:
# numPhotons = 40000  # usually set to evEnergy / phEnergy

###############################################################################
# Set event position
eventX = random.randint(0,barLength)
eventY = random.randint(0,barWidth)
eventZ = random.randint(0,barHeight)

numPhotons = int(numPhotons * quantumEfficiency)
totalCount = 0
arrTimes = [0] * numPhotons

print("Total Number of photons modelled from scintillation (reduced by PMT quantum efficiency): " + str(numPhotons))

# Propogate each photon (initialize counter)
phCount = 1
while phCount <= numPhotons:
    blnTransmitted = False
    blnAtFibreEdge = False
    exception = False
    blnPhotonLost = False
    
    #Initialize photon object in MIDDLE of the 2Dbar. 
    #Energy in eV of scintillation photon from emission peak of 476nm. Origin in bottom left: 0,0,0.
    # arguments: dictPos, energy, barX, barY, barZ, fibreRadius
    ph2D = Photon({'xPos': eventX, 'yPos': eventY, 'zPos': eventZ}, phEnergy, barLength, barWidth, barHeight, fibreRadius)
    
    
    while blnTransmitted == False and exception == False and blnPhotonLost == False:    
        for x in range (1000):
            if x == 999 or blnTransmitted == True:
                blnTransmitted = True
                break
            elif x == 999 or blnPhotonLost == True:
                blnPhotonLost = True
                break
            elif x == 999 or exception == True:
                exception = True
                break
        
            ph2D.movePhoton()

    arrTimes[phCount-1] = ph2D.distanceTravelled/speedOfLight

    
    phCount += 1
    
###############################################################################
# END SIMULATION #
###############################################################################

###############################################################################


###############################################################################
# POST PROCESSING #
###############################################################################
# Post processing on the output array containing photon arrival times, due to variables overflowing.
    
# Remove 0 values for distance travelled due to Lost photons from output
arrTimesProcessed = []
for each in arrTimes:
    if each != 0:
        arrTimesProcessed.append(each)
        
   
# A limiting window of 1 micro second is applied here.

for i in range(0, len(arrTimesProcessed)):
    if arrTimesProcessed[i] > (window / 10**-9):
        arrTimesProcessed[i] = -1    

arrOut = [[-1*peVoltage] * len(arrTimesProcessed)] # , arrTimes]

plt.xlabel('Time (ns)')
plt.ylabel('Approx single PE amplitude (V)')
plt.scatter(arrTimesProcessed, arrOut)
plt.show()

print(str(len(arrTimes) - len(arrTimesProcessed)) + " total photons lost, " + str(len(arrTimesProcessed)) + " remain")
###############################################################################


##############################################################################
# BINNING #
##############################################################################

def create_bins(lower_bound, width, quantity):
    bins = []
    for low in np.arange(lower_bound, max(arrTimesProcessed), width):
        bins.append((round(low,6), round(low+width,6)))
    return bins



def find_bin(value, bins):    
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1


bins = create_bins(lower_bound=0,
                   width=binWidth,
                   quantity=int(max(arrTimesProcessed)))


binned_weights = []
for value in arrTimesProcessed:
    bin_index = find_bin(round(value,1), bins)
    binned_weights.append(bin_index)
 
frequencies = Counter(binned_weights)

arrAmplitude = []
for each in frequencies.values():
    arrAmplitude.append(each * peVoltage)
    
arrBins = []
for each in frequencies.keys():
    arrBins.append(each * binWidth)
 
#plt.xlabel('Bin number')
#plt.ylabel('Photon count')
#plt.bar(frequencies.keys(), frequencies.values())
#plt.show()

plt.xlabel('Time (ns)')
plt.ylabel('Amplitude (V)')
plt.scatter(arrBins, arrAmplitude)
plt.show()

##############################################################################



