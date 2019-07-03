from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from Carbon.Controls import inTriangle
from numpy import average
from scipy.spatial import ConvexHull
from matplotlib.mlab import griddata
import matplotlib.tri as tri
from Carbon.QuickDraw import extend
import pyopencl as cl
import time
import os
from scipy import ndimage

'''
Created on Jan 6, 2017

@author: jihoonHyun
'''


# Opencl C source code for parallel computing

kernel_extension = """

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void ClAddMesh(__global int* nn0,
			__global int* nn1,
			__global float* data,
			__global float* result)
{
	int i = get_global_id(0);
    
    
        //printf(\"%d %d %d \\n\",i, nn0[i]-1, nn1[i]-1);

	for(int j =0; j<16;j++) {
        int index = i*16 + j;
        
        int index0 = nn0[index]-1;
        int index1 = nn1[index]-1;
		result[i] += data[index0] + data[index1];
        //if(i == 0) printf(\"%d %d %.3f %.3f \\n\",index0, index1, data[index0],data[index1]);
	}
	result[i] = result[i]/16.0;
    
    //if(i==0) printf(\"%d %.3f \\n\",i,result[i]);
    

    
}
"""



kernel_reducing = """
    
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
    
__kernel void ClReducing(__global float* data,
                        __global float* node,
                        __global int* elem,
                        __global float* new_data,
                        __global float* minData)
    {
    

        int i = get_global_id(0);
        float thres = 0.5 * minData[0];
        
        for(int k = 0; k <4;k++)
        {
            int index1 = i*4 + k;
            int index2 = (elem[index1]-1)*3;
            float xt = node[index2];
            float yt = node[index2 +1];
            float zt = node[index2 +2];
            
            //printf(\"%d %d %.3f %.3f %.3f\\n\",i,index2/3+1,xt,yt,zt);
            
            if(data[i] > thres) {new_data[i] = 0; break;} //printf(\"1 %d %d %.3f %.3f\\n\",i,index/3+1,oridata,thres);}
            else if(zt < -105*0.8) {new_data[i] =0; break;} //printf(\"2 %d %d %.3f %.3f\\n\",i,index,zt,-105*0.8);}
            else if(zt > 105*0.8) {new_data[i] =0; break;} //printf(\"3 %d %d %.3f %.3f\\n\",i,index,zt,-105*0.8);}
            else if((xt*xt)+(yt*yt) > 14400.0) {new_data[i] =0; break;} //printf(\"4 %d %.3f\\n\",i,index,(xt*xt)+(yt*yt));}
            else if(xt < 140*0.05 && xt > -140*0.05) {new_data[i] =0; break;}// printf(\"4 %d %.3f\\n\",i,index,(xt*xt)+(yt*yt));}
            else{new_data[i] = data[i];}
            //printf(\"%d %d %.3f %.3f\\n\",i,index,node[index], new_data[i], oridata);
            
        }
    }
"""


def get_tpoints(points,elem,node):
    for i in range(len(elem)):
        for k in range(4):
            points[i][0] = node[elem[i][k]-1][0];
            points[i][1] = node[elem[i][k]-1][1];
            points[i][2] = node[elem[i][k]-1][2];


def read_mat(filename):
    return sio.loadmat(filename);
    

if __name__ == '__main__':
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    os.environ['PYOPENCL_CTX'] = '0:1'
    start_time = time.time();
    mat_data_3D = read_mat("data_3D.mat");
    mat_data = read_mat("data.mat");
    
    mat_data_pos = [];
    for i in range(5):
        mat_data_pos.append(read_mat("data_pos"+str(i+1)+".mat"));
    
    elem = np.array(mat_data_3D['elem_grid']);
    node = np.array(mat_data_3D['node_grid']);

    elem = np.ravel(elem);
    node = np.ravel(node);
 

    #rm = np.array(mat_data['RM']);
    fig = plt.figure();
    ax = Axes3D(fig);
    ax.set_xlim3d(-150, 150)
    ax.set_ylim3d(-150, 150)
    ax.set_zlim3d(-100, 100)
    cm = plt.get_cmap("RdYlGn");
    plt.ion();

    #extended mesh part
    extendedMatData = read_mat("large_data_3D.mat");
    large_elem = np.array(extendedMatData['large_elems']);
    large_node = np.array(extendedMatData['large_nodes']);

    nn0 = np.array(extendedMatData['rimg_NN0']);
    nn1 = np.array(extendedMatData['rimg_NN1']);
    ss = len(nn0);
    #nn0 = np.transpose(nn0);
    #nn1 = np.transpose(nn1);
    nn0 = np.ravel(nn0);
    nn1 = np.ravel(nn1);

    load_time = time.time();
    print("load time", load_time- start_time);

    extendedData = np.zeros(ss, dtype = np.float32);

    context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    program_extension = cl.Program(context, kernel_extension).build()
    program_reducing = cl.Program(context, kernel_reducing).build()

    points = np.zeros((len(large_elem),3));

    get_tpoints(points,large_elem,large_node);

    subfigure = [];
    c = 0;
    while(True):
        try:
            plt.cla();
            ax.set_xlim3d(-150, 150)
            ax.set_ylim3d(-150, 150)
            ax.set_zlim3d(-100, 100)
            temp_time1 = time.time();
            
            data = np.array(mat_data_pos[c]['elem_data']);
            
            c = (c+1)%5;
            #data = np.array(mat_data['elem_data']);
            data = np.ravel(data);
            new_data = np.zeros(len(data), dtype = np.float32);
           
            minData = [];
            minData.append(np.min(data));
            data_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf = np.float32(data));
            elem_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY| cl.mem_flags.COPY_HOST_PTR, hostbuf = np.int32(elem));
            node_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY| cl.mem_flags.COPY_HOST_PTR, hostbuf = np.float32(node));
            minData_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY| cl.mem_flags.COPY_HOST_PTR, hostbuf = np.float32(minData));
            new_data_dev = cl.Buffer(context, cl.mem_flags.WRITE_ONLY| cl.mem_flags.COPY_HOST_PTR, hostbuf = np.float32(new_data));

            program_reducing.ClReducing(queue, data.shape, None, data_dev,node_dev,elem_dev,new_data_dev,minData_dev);

            cl.enqueue_read_buffer(queue, new_data_dev, new_data).wait();
            
            data = new_data;
            
            # Opencl routines and objects
            # create device side vectors and copy values from host to device memory
            nn0_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY| cl.mem_flags.COPY_HOST_PTR, hostbuf = np.int32(nn0));
            nn1_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY| cl.mem_flags.COPY_HOST_PTR, hostbuf = np.int32(nn1));
            extendedData_dev = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=extendedData);
            data_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = np.float32(data));
            program_extension.ClAddMesh(queue, extendedData.shape, None, nn0_dev, nn1_dev, data_dev, extendedData_dev);
            cl.enqueue_read_buffer(queue, extendedData_dev, extendedData).wait();
            extend_time = time.time();
            print("extend time", extend_time -temp_time1);
            x1 = [];
            y1 = [];
            z1 = [];
            
            x2 = [];
            y2 = [];
            z2 = [];
            
            
            thres = 0.3;
            extendedDataMin = np.min(extendedData) * thres;
            verts1 = []
            verts2 = []
            
            for i in range(len(extendedData)):
                if extendedData[i] < extendedDataMin:
                    if points[i][0] > 0:
                        verts1.append((points[i][0],points[i][1],points[i][2]));
                    else:
                        verts2.append((points[i][0],points[i][1],points[i][2]));
                            
            addpoint_time = time.time()
            print("Adding points time", addpoint_time-extend_time );


            hull1 = ConvexHull(verts1);
            hull2 = ConvexHull(verts2);
            
            xn1 = [];
            yn1 = [];
            zn1 = [];
            xn2 = [];
            yn2 = [];
            zn2 = [];
            
            
            for i in hull1.vertices:
                xn1.append(verts1[i][0]);
                yn1.append(verts1[i][1]);
                zn1.append(verts1[i][2]);
            for i in hull2.vertices:
                xn2.append(verts2[i][0]);
                yn2.append(verts2[i][1]);
                zn2.append(verts2[i][2]);
            
            verts1 = np.transpose(verts1);
            verts2 = np.transpose(verts2);
            
            
            #ax.scatter(xn1, yn1, zn1);
            #ax.scatter(xn2, yn2, zn2);
            tri1 = tri.Triangulation(verts1[0], verts1[1], triangles=hull1.simplices);
            tri2 = tri.Triangulation(verts2[0], verts2[1], triangles=hull2.simplices);
        
            
            ax.plot_trisurf(tri1, verts1[2], color = "grey", alpha = 0.5, linewidth = 0,  antialiased=True);
            ax.plot_trisurf(tri2, verts2[2], color = "grey", linewidth = 0,  alpha = 0.5, antialiased=True);

            #ax.scatter(xn1_mesh, yn1_mesh, zn1_mesh);
            #ax.plot_surface(xn1_mesh, yn1_mesh, zn1_mesh, rstride=1, cstride=1, linewidth=1, antialiased=True)
            #ax.plot_surface(xn12,yn12,zn12,rstride=1, cstride=1, linewidth=0, antialiased=False)
            
            end_time = time.time();
            
            
            print("creating mesh time", end_time-addpoint_time);
            print "cycle time" ,end_time-temp_time1;
            print "total time" ,end_time-start_time;
            plt.pause(0.01);
        except(KeyboardInterrupt):
            plt.ioff();
            break;



    
    
