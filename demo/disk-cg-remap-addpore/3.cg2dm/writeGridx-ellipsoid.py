import numpy as np
#import minpy.numpy as np
import time
from multiprocessing import Pool
import sys
from scipy.spatial.transform import Rotation as R

# User input
prefix = 'add_voro_quat_scale5.78_202002_4x4x4.1210000'
atomnum = 64
gridNum = 128
shape_a = 57.8
shape_b = 57.8
shape_c = 5.78

if(len(sys.argv)>=3):
    prefix = sys.argv[1]
    atomnum = int(sys.argv[2])
    gridNum = int(sys.argv[3])
#minimumClusterWeight = 100
fileName = prefix + '.lammpstrj'
outName  = prefix + '-grid%d.xyz'%(gridNum)

count = 0
for count, line in enumerate(open(fileName, 'rU')):
    count += 1

with open(fileName, 'r') as f:
    contents = f.readlines()

print(type(contents))

index1 = count - atomnum -4
index2 = count - atomnum -1
index3 = count - atomnum
index4 = count +1

contents_bc = contents[index1:index2]
bc = [ [0.0]*2 for i in range(len(contents_bc))]

for i in range(len(contents_bc)):
    currentLine = contents_bc[i].split()
    bc[i][0] = float("{:.1f}".format(float(currentLine[0])))
    bc[i][1] = float("{:.1f}".format(float(currentLine[1])))

xlo = bc[0][0]
xhi = bc[0][1]
ylo = bc[1][0]
yhi = bc[1][1]
zlo = bc[2][0]
zhi = bc[2][1]
boxSize = np.array([xhi-xlo, yhi-ylo, zhi-zlo])

contents_atom = contents[index3:index4]
atomData = [ [0.0]*11 for i in range(len(contents_atom))]

for i in range(len(contents_atom)):
    currentLine = contents_atom[i].split()
    atomData[i][0] = float(currentLine[0]) # atom ID
    atomData[i][1] = float(currentLine[2]) # pos x 
    atomData[i][2] = float(currentLine[3]) # pos y
    atomData[i][3] = float(currentLine[4]) # pos z
    atomData[i][4] = float(currentLine[6]) # quat x
    atomData[i][5] = float(currentLine[7]) # quat y
    atomData[i][6] = float(currentLine[8]) # quat z
    atomData[i][7] = float(currentLine[5]) # quat w
    #atomData[i][8] = float(currentLine[9]) # shape a
    #atomData[i][9] = float(currentLine[10]) # shape b
    #atomData[i][10] = float(currentLine[11]) # shape c
    atomData[i][8] = shape_a # shape a
    atomData[i][9] = shape_b # shape b
    atomData[i][10] = shape_c # shape c

atomData.sort(key=lambda x: x[0])
atomData = np.array(atomData)

maxShapeRadius = np.max(atomData[:,8:11])

atomDataXlo = np.copy(atomData[np.where( atomData[:,1] < (xlo + maxShapeRadius) )[0],:])
atomDataXhi = np.copy(atomData[np.where( atomData[:,1] > (xhi - maxShapeRadius) )[0],:])
atomDataYlo = np.copy(atomData[np.where( atomData[:,2] < (ylo + maxShapeRadius) )[0],:])
atomDataYhi = np.copy(atomData[np.where( atomData[:,2] > (yhi - maxShapeRadius) )[0],:])
atomDataZlo = np.copy(atomData[np.where( atomData[:,3] < (zlo + maxShapeRadius) )[0],:])
atomDataZhi = np.copy(atomData[np.where( atomData[:,3] > (zhi - maxShapeRadius) )[0],:])

atomDataXlo[:,1] += (xhi-xlo)
atomDataXhi[:,1] -= (xhi-xlo)
atomDataYlo[:,2] += (yhi-ylo)
atomDataYhi[:,2] -= (yhi-ylo)
atomDataZlo[:,3] += (zhi-zlo)
atomDataZhi[:,3] -= (zhi-zlo)

atomData = np.concatenate((atomData, atomDataXlo, atomDataXhi, atomDataYlo, atomDataYhi, atomDataZlo, atomDataZhi), axis=0)
print(len(atomData))

xNum = gridNum
yNum = gridNum
zNum = gridNum
dx = (xhi-xlo)/xNum
dy = (yhi-ylo)/yNum
dz = (zhi-zlo)/zNum
print(xNum,yNum,zNum)       

def find_closest_points8000(pos, pos2, shape, quat):
    #pos2: 1x3 vector, absolute xyz coordiate of probe
    #pos: nx3 vector, center of all disks
    #shape: nx3 vector, radius of two disks
    #quat: nx4 vector, orientation of the two disks
    atomNum = len(pos)
    # Radius of disk
    r1 = shape[:,0] - shape[:,2]
    
    # 3x3 rotation matrix of disk
    a1 = R.from_quat(quat).as_matrix()
    
    # vec from pos1 to pos2
    R12 = -1*(np.array(pos) - np.array(pos2))

    # norm vec of the disk
    #ZVec = np.array([0, 0, 1]).dot(a1)
    ZVec = np.dot(a1, np.array([0, 0, 1]))
    # dot product of R12 and ZVec
    ZVecProjVecLength = np.sum(np.multiply(R12,ZVec),axis=1).reshape(atomNum,1)
    # projection of R12 on disk1 (originated at pos1)
    XYPlaneProjVec = R12 - np.multiply(ZVecProjVecLength,ZVec)

    # check if it is out of the disk and scale the projected vec
    dist2DiskCenter = np.sqrt( np.sum( XYPlaneProjVec**2, axis=1 ) )
    selectedIndex = np.argwhere(dist2DiskCenter>r1)
    selectedXYProj = XYPlaneProjVec[selectedIndex[:,0],:]
    scaleFactor = np.divide(r1[selectedIndex], dist2DiskCenter[selectedIndex])
    XYPlaneProjVec[selectedIndex[:,0],:] = np.multiply( selectedXYProj, scaleFactor )

    # transition to global coordinate
    pointOnDisk = np.array(pos) + XYPlaneProjVec
    return pointOnDisk

def closestDistancePeriodic(xyz, atomData = atomData, boxSize = boxSize):
    #print(xyz)
    atomXYZ = atomData[:,1:4]
    atomShape = atomData[:,8:11]
    atomQuat = atomData[:,4:8]
    atomthick = atomShape[:,2]
    atompos = np.zeros((len(atomData),3))
    #for m in range(8000):
    #    atompos[m,:] = find_closest_points2(atomXYZ[m,:], xyz, atomShape[m,:], atomQuat[m,:])
    atompos = find_closest_points8000(atomXYZ, xyz, atomShape, atomQuat)
    for i in range(3):
        delta = np.abs(atompos[:,i] - xyz[i])
        #delta[delta>(0.5*boxSize[i])] -= boxSize[i]
        if(i==0):
            diffSqrTotal = delta**2
        else:
            diffSqrTotal+= delta**2
    diffSqrTotalSqrt = np.sqrt(diffSqrTotal) 
    allDist = diffSqrTotalSqrt - atomthick
    minDist = allDist.min()
    return xyz[0], xyz[1], xyz[2], minDist 

tic = time.time()
xyzIter = [] 
for k in range(zNum):
        for j in range(yNum):
            #print(j)
            for i in range(xNum):
                index = k*(yNum)*(xNum) + j*(xNum) + i + 1
                coordX = ( i + 0.5 )*dx + xlo  # x coordinate
                coordY = ( j + 0.5 )*dy + ylo  # y coordinate
                coordZ = ( k + 0.5 )*dz + zlo  # z coordinate
                xyzIter.append([coordX, coordY, coordZ])

#test
#print(closestDistancePeriodic([0.5,0.5,0.5]))
#for i in range(100):
#    print(closestDistancePeriodic([50.5,50.5,50.5]))
#    tok = time.time()
#    print("Time consumed: %f"%(tok - tic))
#    tic = time.time()
#exit()

if __name__ == '__main__':
    with Pool(4) as p:
        #results = p.map(closestDistance, xyzIter)
        results = p.map(closestDistancePeriodic, xyzIter)
    #for i in range(len(results)):
    #    ii = results[i][0]
    #    jj = results[i][1]
    #    kk = results[i][2]
    #    grid[ii][jj][kk] = results[i][3]
    #writeGrid2lmpData2(grid, box, outName2)
    tok = time.time()
    print("Time consumed: %f"%(tok - tic))

    with open(outName, 'w') as output:
        line2Write = "%d\n" % (xNum*yNum*zNum)
        output.write(line2Write)
        line2Write = "Lattice=\"%f 0.0 0.0 0.0 %f 0.0 0.0 0.0 %f\" Properties=pos:R:3:CDist:R:1\n"%(xhi-xlo,yhi-ylo,zhi-zlo)
        output.write(line2Write)
        for info in results:
            line2Write = "%.1f %.1f %.1f %f\n" % (info[0], info[1], info[2], info[3])
            output.write(line2Write)    
