#!/usr/bin/python
"""
    K40 Whisperer

    Copyright (C) <2018>  <Scorch>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import sys
from math import *
from egv import egv
from nano_library import K40_CLASS
from dxf import DXF_CLASS
from svg_reader import SVG_READER
from svg_reader import SVG_TEXT_EXCEPTION
from g_code_library import G_Code_Rip
from interpolate import interpolate

import traceback
DEBUG = False

VERSION = sys.version_info[0]
if VERSION < 3 and sys.version_info[1] < 6:
    def next(item):
        return item.next()
    
import os

from PIL import Image
from PIL import ImageOps
try:
    Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)
except:
    pass
try:
    from PIL import ImageTk
    from PIL import _imaging
except:
    pass #Don't worry everything will still work

QUIET = False

class ECoord():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.image      = None
        self.reset_path()

    def reset_path(self):
        self.ecoords    = []
        self.len        = 0
        self.move       = 0
        self.sorted     = False
        self.bounds     = (0,0,0,0)
        self.gcode_time = 0        

    def make_ecoords(self,coords,scale=1):
        self.reset()
        self.len  = 0
        self.move = 0
        
        xmax, ymax = -1e10, -1e10
        xmin, ymin =  1e10,  1e10
        self.ecoords=[]
        Acc=.001
        oldx = oldy = -99990.0
        first_stroke = True
        loop=0
        for line in coords:
            XY = line
            x1 = XY[0]*scale
            y1 = XY[1]*scale
            x2 = XY[2]*scale
            y2 = XY[3]*scale
            dxline= x2-x1
            dyline= y2-y1
            len_line=sqrt(dxline*dxline + dyline*dyline)
            
            dx = oldx - x1
            dy = oldy - y1
            dist   = sqrt(dx*dx + dy*dy)
            # check and see if we need to move to a new discontinuous start point
            if (dist > Acc) or first_stroke:
                loop = loop+1
                self.ecoords.append([x1,y1,loop])
                if not first_stroke:
                    self.move = self.move + dist
                first_stroke = False
                
            self.len = self.len + len_line
            self.ecoords.append([x2,y2,loop])
            oldx, oldy = x2, y2
            xmax=max(xmax,x1,x2)
            ymax=max(ymax,y1,y2)
            xmin=min(xmin,x1,x2)
            ymin=min(ymin,y1,y2)
        self.bounds = (xmin,xmax,ymin,ymax)

    def set_ecoords(self,ecoords,data_sorted=False):
        self.ecoords = ecoords
        self.computeEcoordsLen()
        self.data_sorted=data_sorted

    def set_image(self,PIL_image):
        self.image = PIL_image

    def computeEcoordsLen(self):
        xmax, ymax = -1e10, -1e10
        xmin, ymin =  1e10,  1e10
        
        if self.ecoords == [] : 
            return
        on = 0
        move = 0
        time = 0
        for i in range(2,len(self.ecoords)):
            x1 = self.ecoords[i-1][0]
            y1 = self.ecoords[i-1][1]
            x2 = self.ecoords[i][0]
            y2 = self.ecoords[i][1]
            loop      = self.ecoords[i  ][2]
            loop_last = self.ecoords[i-1][2]
            
            xmax=max(xmax,x1,x2)
            ymax=max(ymax,y1,y2)
            xmin=min(xmin,x1,x2)
            ymin=min(ymin,y1,y2)
            
            dx = x2-x1
            dy = y2-y1
            dist = sqrt(dx*dx + dy*dy)
            
            if len(self.ecoords[i]) > 3:
                feed = self.ecoords[i][3]
                time = time + dist/feed*60
                
            if loop == loop_last:
                on   = on + dist 
            else:
                move = move + dist
            #print "on= ",on," move= ",move
        self.bounds = (xmin,xmax,ymin,ymax)
        self.len = on
        self.move = move
        self.gcode_time = time
    
################################################################################
class WhispererCore():
    def __init__(self, config={}, update_gui=None):
        self.k40 = None
        self.stop_flag = [False]
        self.update_gui = update_gui if update_gui else self.NOP
        self.config = config
        print("config init", config)

    def NOP(self, str):
        # no operation
        pass

    def resetPath(self):
        self.RengData  = ECoord()
        self.VengData  = ECoord()
        self.VcutData  = ECoord()
        self.GcodeData = ECoord()
        self.Design_bounds = (0,0,0,0)
        self.pos_offset=[0.0,0.0]
        
################################################################################
    def getWorkspace(self):
        return (0.0, self.config.get("LaserXsize", 0.0), self.config.get("LaserYsize", 0.0), 0.0)


    ################################################################################


##    def computeAccurateVeng(self):
##        self.update_gui("Optimize vector engrave.") 
##        self.VengData.set_ecoords(self.optimize_paths(self.VengData.ecoords),data_sorted=True)
##        self.refreshTime()
##            
##    def computeAccurateVcut(self):
##        self.update_gui("Optimize vector cut.") 
##        self.VcutData.set_ecoords(self.optimize_paths(self.VcutData.ecoords),data_sorted=True)
##        self.refreshTime()
##
##    def computeAccurateReng(self):
##        self.update_gui("Calculating Raster engrave.")
##        if self.RengData.image != None:        
##            if self.RengData.ecoords == []:
##                self.make_raster_coords()
##        self.RengData.sorted = True 
##        self.refreshTime()

    #######################################################################


    def make_ecoords(self,coords,scale=1):
        xmax, ymax = -1e10, -1e10
        xmin, ymin =  1e10,  1e10
        ecoords=[]
        Acc=.001
        oldx = oldy = -99990.0
        first_stroke = True
        loop=0
        for line in coords:
            XY = line
            x1 = XY[0]*scale
            y1 = XY[1]*scale
            x2 = XY[2]*scale
            y2 = XY[3]*scale
            dx = oldx - x1
            dy = oldy - y1
            dist = sqrt(dx*dx + dy*dy)
            # check and see if we need to move to a new discontinuous start point
            if (dist > Acc) or first_stroke:
                loop = loop+1
                first_stroke = False
                ecoords.append([x1,y1,loop])
            ecoords.append([x2,y2,loop])
            oldx, oldy = x2, y2
            xmax=max(xmax,x1,x2)
            ymax=max(ymax,y1,y2)
            xmin=min(xmin,x1,x2)
            ymin=min(ymin,y1,y2)
        bounds = (xmin,xmax,ymin,ymax)
        return ecoords,bounds

##    def convert_greyscale(self,image):
##        image = image.convert('RGB')
##        x_lim, y_lim = image.size
##        grey = Image.new("L", image.size, color=255)
##        
##        pixel = image.load()
##        grey_pixel = grey.load()
##        
##        for y in range(1, y_lim):
##            self.update_gui("Converting to greyscale: %.1f %%" %( (100.0*y)/y_lim ) )
##            self.master.update()
##            for x in range(1, x_lim):
##                value = pixel[x, y]
##                grey_pixel[x,y] = int( value[0]*.333 + value[1]*.333 +value[2]*.333 )
##                #grey_pixel[x,y] = int( value[0]*.210 + value[1]*.720 +value[2]*.007 )
##                grey_pixel[x,y] = int( value[0]*.299 + value[1]*.587 +value[2]*.114 )
##
##        grey.save("adjusted_grey.png","PNG")
##        return grey

    
    #####################################################################
    def make_raster_coords(self, mirror=False, rotate=False, halftone = {}):
            ecoords=[]
            if (self.RengData.image != None):
                cutoff=128
                image_temp = self.RengData.image.convert("L")

                if mirror:
                    image_temp = ImageOps.mirror(image_temp)

                if rotate:
                    #image_temp = image_temp.rotate(90,expand=True)
                    image_temp = self.rotate_raster(image_temp)

                Xscale = float(self.config.get("LaserXscale", 1))
                Yscale = float(self.config.get("LaserYscale", 1))
                if Xscale != 1.0 or Yscale != 1.0:
                    wim,him = image_temp.size
                    nw = int(wim*Xscale)
                    nh = int(him*Yscale)
                    image_temp = image_temp.resize((nw,nh))



                npixels = int( round(halftone.get("ht_size_mils", 0),1) )
                if npixels > 0:
                    #start = time()
                    wim,him = image_temp.size
                    # Convert to Halftoning and save
                    nw=int(wim / npixels)
                    nh=int(him / npixels)
                    image_temp = image_temp.resize((nw,nh))

                    M1 = float(halftone.get("bezier_M1"))
                    M2 = float(halftone.get("bezier_M2"))
                    w = float(halftone.get("bezier_weight"))

                    image_temp = self.convert_halftoning(image_temp, M1, M2, w)
                    image_temp = image_temp.resize((wim,him))
                    #print time()-start
                    
                if DEBUG:
                    image_name = os.path.expanduser("~")+"/IMAGE.png"
                    image_temp.save(image_name,"PNG")

                Reng_np = image_temp.load()
                wim,him = image_temp.size
                #######################################
                x=0
                y=0
                loop=1
                
                for i in range(0,him,self.config.get("raster_step_1000in",1)):
                    if i%100 ==0:
                        self.update_gui("Raster Engraving: Creating Scan Lines: %.1f %%" %( (100.0*i)/him ) )
                    if self.stop_flag[0]==True:
                        raise StandardError("Action stopped by User.")
                    line = []
                    cnt=1
                    for j in range(1,wim):
                        if (Reng_np[j,i] == Reng_np[j-1,i]):
                            cnt = cnt+1
                        else:
                            laser = "U" if Reng_np[j-1,i] > cutoff else "D"
                            line.append((cnt,laser))
                            cnt=1
                    laser = "U" if Reng_np[j-1,i] > cutoff else "D"
                    line.append((cnt,laser))
                    
                    y=(him-i)/1000.0
                    x=0
                    rng = range(0,len(line),1)
                        
                    for i in rng:
                        seg = line[i]
                        delta = seg[0]/1000.0
                        if seg[1]=="D":
                            loop=loop+1
                            ecoords.append([x      ,y,loop])
                            ecoords.append([x+delta,y,loop])
                        x = x + delta
                        
            if ecoords!=[]:
                self.RengData.set_ecoords(ecoords,data_sorted=True)
    #######################################################################


    def rotate_raster(self,image_in):
        wim,him = image_in.size
        im_rotated = Image.new("L", (him, wim), "white")

        image_in_np   = image_in.load()
        im_rotated_np = im_rotated.load()
        
        for i in range(1,him):
            for j in range(1,wim):
                im_rotated_np[i,wim-j] = image_in_np[j,i]
        return im_rotated
    
    '''This Example opens an Image and transform the image into halftone.  -Isai B. Cicourel'''
    # Create a Half-tone version of the image
    def convert_halftoning(self, image, M1, M2, w):
        image = image.convert('L')
        x_lim, y_lim = image.size
        pixel = image.load()

        if w > 0:
            x,y = self.generate_bezier(M1,M2,w)

            interp = interpolate(x, y) # Set up interpolate class
            val_map=[]
            # Map Bezier Curve to values between 0 and 255
            for val in range(0,256):
                val_out = int(round(interp[val])) # Get the interpolated value at each value
                val_map.append(val_out)
            # Adjust image
            for y in range(1, y_lim):
                self.update_gui("Raster Engraving: Adjusting Image Darkness: %.1f %%" %( (100.0*y)/y_lim ) )
                for x in range(1, x_lim):
                    pixel[x, y] = val_map[ pixel[x, y] ]

        self.update_gui("Raster Engraving: Creating Halftone Image." )
        image = image.convert('1')
        return image

    def generate_bezier(self,M1,M2,w,n=100):
        if (M1==M2):
            x1=0
            y1=0
        else:
            x1 = 255*(1-M2)/(M1-M2)
            y1 = M1*x1
        x=[]
        y=[]
        # Calculate Bezier Curve
        for step in range(0,n+1):
            t    = float(step)/float(n)
            Ct   = 1 / ( pow(1-t,2)+2*(1-t)*t*w+pow(t,2) )
            x.append( Ct*( 2*(1-t)*t*w*x1+pow(t,2)*255) )
            y.append( Ct*( 2*(1-t)*t*w*y1+pow(t,2)*255) )
        return x,y

    ################################################################################
    def Sort_Paths(self,ecoords,i_loop=2):
        ##########################
        ###   find loop ends   ###
        ##########################
        Lbeg=[]
        Lend=[]
        if len(ecoords)>0:
            Lbeg.append(0)
            loop_old=ecoords[0][i_loop]
            for i in range(1,len(ecoords)):
                loop = ecoords[i][i_loop]
                if loop != loop_old:
                    Lbeg.append(i)
                    Lend.append(i-1)
                loop_old=loop
            Lend.append(i)

        #######################################################
        # Find new order based on distance to next beg or end #
        #######################################################
        order_out = []
        use_beg=0
        if len(ecoords)>0:
            order_out.append([Lbeg[0],Lend[0]])
        inext = 0
        total=len(Lbeg)
        for i in range(total-1):
            if use_beg==1:
                ii=Lbeg.pop(inext)
                Lend.pop(inext)
            else:
                ii=Lend.pop(inext)
                Lbeg.pop(inext)

            Xcur = ecoords[ii][0]
            Ycur = ecoords[ii][1]

            dx = Xcur - ecoords[ Lbeg[0] ][0]
            dy = Ycur - ecoords[ Lbeg[0] ][1]
            min_dist = dx*dx + dy*dy

            dxe = Xcur - ecoords[ Lend[0] ][0]
            dye = Ycur - ecoords[ Lend[0] ][1]
            min_diste = dxe*dxe + dye*dye

            inext=0
            inexte=0
            for j in range(1,len(Lbeg)):
                dx = Xcur - ecoords[ Lbeg[j] ][0]
                dy = Ycur - ecoords[ Lbeg[j] ][1]
                dist = dx*dx + dy*dy
                if dist < min_dist:
                    min_dist=dist
                    inext=j
                ###
                dxe = Xcur - ecoords[ Lend[j] ][0]
                dye = Ycur - ecoords[ Lend[j] ][1]
                diste = dxe*dxe + dye*dye
                if diste < min_diste:
                    min_diste=diste
                    inexte=j
                ###
            if min_diste < min_dist:
                inext=inexte
                order_out.append([Lend[inexte],Lbeg[inexte]])
                use_beg=1
            else:
                order_out.append([Lbeg[inext],Lend[inext]])
                use_beg=0
        ###########################################################
        return order_out
    
    #####################################################
    # determine if a point is inside a given polygon or not
    # Polygon is a list of (x,y) pairs.
    # http://www.ariel.com.au/a/python-point-int-poly.html
    #####################################################
    def point_inside_polygon(self,x,y,poly):
        n = len(poly)
        inside = -1
        p1x = poly[0][0]
        p1y = poly[0][1]
        for i in range(n+1):
            p2x = poly[i%n][0]
            p2y = poly[i%n][1]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = inside * -1
            p1x,p1y = p2x,p2y

        return inside

    def optimize_paths(self,ecoords):
        order_out = self.Sort_Paths(ecoords)
        lastx=-999
        lasty=-999
        Acc=0.004
        cuts=[]

        for line in order_out:
            temp=line
            if temp[0] > temp[1]:
                step = -1
            else:
                step = 1

            loop_old = -1
            
            for i in range(temp[0],temp[1]+step,step):
                x1   = ecoords[i][0]
                y1   = ecoords[i][1]
                loop = ecoords[i][2]
                # check and see if we need to move to a new discontinuous start point
                if (loop != loop_old):
                    dx = x1-lastx
                    dy = y1-lasty
                    dist = sqrt(dx*dx + dy*dy)
                    if dist > Acc:
                        cuts.append([[x1,y1]])
                    else:
                        cuts[-1].append([x1,y1])
                else:
                    cuts[-1].append([x1,y1])
                lastx = x1
                lasty = y1
                loop_old = loop
        #####################################################
        # For each loop determine if other loops are inside #
        #####################################################
        Nloops=len(cuts)
        self.LoopTree=[]
        for iloop in range(Nloops):
            self.LoopTree.append([])
            ipoly = cuts[iloop]
            ## Check points in other loops (could just check one) ##
            if ipoly != []:
                for jloop in range(Nloops):
                    if jloop != iloop:
                        inside = 0
                        inside = inside + self.point_inside_polygon(cuts[jloop][0][0],cuts[jloop][0][1],ipoly)
                        if inside > 0:
                            self.LoopTree[iloop].append(jloop)
        #####################################################
        for i in range(Nloops):
            lns=[]
            lns.append(i)
            self.remove_self_references(lns,self.LoopTree[i])

        self.order=[]
        self.loops = range(Nloops)
        for i in range(Nloops):
            if self.LoopTree[i]!=[]:
                self.addlist(self.LoopTree[i])
                self.LoopTree[i]=[]
            if self.loops[i]!=[]:
                self.order.append(self.loops[i])
                self.loops[i]=[]
        ecoords_out = []
        for i in self.order:
            line = cuts[i]
            for coord in line:
                ecoords_out.append([coord[0],coord[1],i])
        return ecoords_out
            
    def remove_self_references(self,loop_numbers,loops):
        for i in range(0,len(loops)):
            for j in range(0,len(loop_numbers)):
                if loops[i]==loop_numbers[j]:
                    loops.pop(i)
                    return
            if self.LoopTree[loops[i]]!=[]:
                loop_numbers.append(loops[i])
                self.remove_self_references(loop_numbers,self.LoopTree[loops[i]])

    def addlist(self,list):
        for i in list:
            try: #this try/except is a bad hack fix to a recursion error. It should be fixed properly later.
                if self.LoopTree[i]!=[]:
                    self.addlist(self.LoopTree[i]) #too many recursions here causes cmp error
                    self.LoopTree[i]=[]
            except:
                pass
            if self.loops[i]!=[]:
                self.order.append(self.loops[i])
                self.loops[i]=[]


    def mirror_vector_coords(self,coords):
        xmin = self.Design_bounds[0]
        xmax = self.Design_bounds[1]
        coords_mirror=[]
        
        for i in range(len(coords)):
            coords_mirror.append(coords[i][:])
            coords_mirror[i][0]=xmin+xmax-coords_mirror[i][0]

        return coords_mirror

    def rotate_vector_coords(self, coords):
        xmin = self.Design_bounds[0]
        xmax = self.Design_bounds[1]
        coords_rotate = []

        for i in range(len(coords)):
            coords_rotate.append(coords[i][:])
            x = coords_rotate[i][0]
            y = coords_rotate[i][1]
            coords_rotate[i][0] = -y
            coords_rotate[i][1] = x

        return coords_rotate

    def scale_vector_coords(self,coords,startx,starty):
        coords_scale=[]
        scaleX = self.config.get("LaserXscale", 1)
        scaleY = self.config.get("LaserYscale", 1)
        if scaleX != 1.0 or scaleY != 1.0:
            for i in range(len(coords)):
                coords_scale.append(coords[i][:])
                x = coords_scale[i][0]
                y = coords_scale[i][1]
                coords_scale[i][0] = x*scaleX
                coords_scale[i][1] = y*scaleY
            scaled_startx = startx*scaleX
            scaled_starty = starty*scaleY
        else:
            coords_scale = coords
            scaled_startx = startx
            scaled_starty = starty

        return coords_scale,scaled_startx,scaled_starty
  
    def send_data(self,operation_type=None):
        print("config", self.config)    # TODO
        if self.k40 == None:
            raise StandardError("Laser Cutter is not Initialized...")
        
        try:
            if self.config.get("inputCSYS") and self.RengData.image == None:
                xmin,xmax,ymin,ymax = 0.0,0.0,0.0,0.0
            else:
                xmin,xmax,ymin,ymax = self.Get_Design_Bounds() # TODO

            startx = xmin
            starty = ymax

            if self.config.get("HomeUR"):
                FlipXoffset = abs(xmax-xmin)
                if self.config.get("rotate"):
                    startx = -xmin
            else:
                FlipXoffset = 0
            
            data=[]
            egv_inst = egv(target=lambda s:data.append(s))
            
            if (operation_type=="Vector_Cut") and  (self.VcutData.ecoords!=[]):
                num_passes = int(self.config.get("Vcut_passes", 1))
                Feed_Rate = float(self.config.get("Vcut_feed"))
                self.update_gui("Vector Cut: Determining Cut Order....")
                if not self.VcutData.sorted and self.config.get("inside_first"):
                    self.VcutData.set_ecoords(self.optimize_paths(self.VcutData.ecoords),data_sorted=True)
                self.update_gui("Generating EGV data...")
                

                Vcut_coords = self.VcutData.ecoords
                if self.config.get("mirror"):
                    Vcut_coords = self.mirror_vector_coords(Vcut_coords)
                if self.config.get("rotate"):
                    Vcut_coords = self.rotate_vector_coords(Vcut_coords)

                Vcut_coords,startx,starty = self.scale_vector_coords(Vcut_coords,startx,starty)
                    
                egv_inst.make_egv_data(
                                                Vcut_coords,                      \
                                                startX=startx,                    \
                                                startY=starty,                    \
                                                Feed = Feed_Rate,                 \
                                                board_name=self.config.get("board_name"), \
                                                Raster_step = 0,                  \
                                                update_gui=self.update_gui,       \
                                                stop_calc=self.stop_flag,              \
                                                FlipXoffset=FlipXoffset
                                                )


            if (operation_type=="Vector_Eng") and  (self.VengData.ecoords!=[]):
                num_passes = int(self.config.get("Veng_passes",1))
                Feed_Rate = float(self.config.get("Veng_feed"))
                self.update_gui("Vector Engrave: Determining Cut Order....")
                if not self.VengData.sorted and self.config.get("inside_first"):
                    self.VengData.set_ecoords(self.optimize_paths(self.VengData.ecoords),data_sorted=True)
                self.update_gui("Generating EGV data...")

                Veng_coords = self.VengData.ecoords
                if self.config.get("mirror"):
                    Veng_coords = self.mirror_vector_coords(Veng_coords)
                if self.config.get("rotate"):
                    Veng_coords = self.rotate_vector_coords(Veng_coords)

                Veng_coords,startx,starty = self.scale_vector_coords(Veng_coords,startx,starty)
                    
                egv_inst.make_egv_data(
                                                Veng_coords,                      \
                                                startX=startx,                    \
                                                startY=starty,                    \
                                                Feed = Feed_Rate,                 \
                                                board_name=self.config.get("board_name"), \
                                                Raster_step = 0,                  \
                                                update_gui=self.update_gui,       \
                                                stop_calc=self.stop_flag,              \
                                                FlipXoffset=FlipXoffset
                                                )

            if (operation_type=="Raster_Eng") and  (self.RengData.ecoords!=[]):
                num_passes = int(self.config.get("Reng_passes",1))
                Feed_Rate = float(self.config.get("Reng_feed"))
                Raster_step = int(self.config.get("raster_step_1000in",1))
                if not self.config.get("engraveUP"):
                    Raster_step = -Raster_step
                    
                raster_startx = 0
                raster_starty = float(self.config.get("LaserYscale", 1))*starty

                self.update_gui("Generating EGV data...")
                egv_inst.make_egv_data(
                                                self.RengData.ecoords,            \
                                                startX=raster_startx,             \
                                                startY=raster_starty,             \
                                                Feed = Feed_Rate,                 \
                                                board_name=self.config.get("board_name"), \
                                                Raster_step = Raster_step,        \
                                                update_gui=self.update_gui,       \
                                                stop_calc=self.stop_flag,              \
                                                FlipXoffset=FlipXoffset
                                                )
                
                self.Reng=[]

            if (operation_type=="Gcode_Cut") and (self.GcodeData!=[]):
                num_passes = int(self.config.get("Gcde_passes",1))
                self.update_gui("Generating EGV data...")

                Gcode_coords = self.GcodeData.ecoords
                if self.config.get("mirror"):
                    Gcode_coords = self.mirror_vector_coords(Gcode_coords)
                if self.config.get("rotate"):
                    Gcode_coords = self.rotate_vector_coords(Gcode_coords)

                Gcode_coords,startx,starty = self.scale_vector_coords(Gcode_coords,startx,starty)
                
                egv_inst.make_egv_data(
                                                Gcode_coords,                     \
                                                startX=startx,                    \
                                                startY=starty,                    \
                                                Feed = None,                      \
                                                board_name=self.config.get("board_name"), \
                                                Raster_step = 0,                  \
                                                update_gui=self.update_gui,       \
                                                stop_calc=self.stop_flag,              \
                                                FlipXoffset=FlipXoffset
                                                )
                
            self.send_egv_data(data, num_passes)
        except MemoryError as e:
            raise StandardError("Memory Error:  Out of Memory.")


    def send_egv_data(self,data,num_passes=1):
        if self.k40 != None:
            self.k40.timeout       = int(self.config.get("t_timeout", 100))
            self.k40.n_timeouts    = int(self.config.get("n_timeouts", 10))
            self.k40.send_data(data,self.update_gui, self.stop_flag, num_passes, self.config.get("pre_pr_crc"))
        if DEBUG:
            print "Saving Data to File...."
            self.write_egv_to_file(data)
        
    ##########################################################################
    ##########################################################################
    def write_egv_to_file(self,data):
        try:
            fname = os.path.expanduser("~")+"/EGV_DATA.EGV"
            fout = open(fname,'w')
        except:
            self.update_gui("Unable to open file for writing: %s" %(fname))
            return
        for char_val in data:
            char = chr(char_val)
            if char == "N":
                fout.write("\n")
                fout.write("%s" %(char))
            elif char == "E":
                fout.write("%s" %(char))
                fout.write("\n")
            else:
                fout.write("%s" %(char))
        fout.write("\n")
        fout.close
        
    def home(self):
        if self.k40 != None:
            self.k40.home_position()
        self.laserX  = 0.0
        self.laserY  = 0.0
        self.pos_offset = [0.0,0.0]

    def reset(self):
        if self.k40 != None:
            self.k40.reset_usb()
            
    def stop(self):
        self.stop_flag[0]=True

    def release(self):
        if self.k40 != None:
            try:
                self.k40.release_usb()
            except:
                self.k40=None
                raise
        
    def initLaser(self,init_home=True):
        self.stop_flag[0]=False
        self.release()
        self.k40=K40_CLASS()
        try:
            self.k40.initialize_device()
            self.k40.say_hello()
            if init_home:
                self.home()
            else:
                self.unlock()
            
        except:
            self.k40=None
            raise

    def unlock(self):
        if self.k40 != None:
            self.k40.unlock_rail()

    def rapid_move(self,dxmils,dymils):
        if self.k40 != None:
            scaleX = float(self.config.get("LaserXscale", 1))
            scaleY = float(self.config.get("LaserXscale", 1))
            dxmils = int(round(dxmils * scaleX))
            dymils = int(round(dymils * scaleY))
            self.k40.rapid_move(dxmils,dymils)
