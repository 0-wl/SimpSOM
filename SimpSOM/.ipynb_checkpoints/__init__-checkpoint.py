"""
SimpSOM (Simple Self-Organizing Maps) v1.4
F. Comitani @2017 
F. Comitani @2018 
F. Comitani @2019 
O. Pellicer @2020
 
A lightweight python library for Kohonen Self-Organising Maps (SOM).
"""

from __future__ import print_function

import sys
import numpy as np
import os, errno

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('../SimpSOM/')
from SimpSOM import hexagons as hx
from SimpSOM import densityPeak as dp
from SimpSOM import qualityThreshold as qt

from sklearn.decomposition import PCA
from sklearn import cluster

#from joblib import Parallel, delayed

class somNet:
    """ Kohonen SOM Network class. """

    def __init__(self, netHeight, netWidth, data, loadFile=None, PCI=0, PBC=0, n_jobs=-1,
                 plot_dpi=70, plot_path='./', plot_show=True, plot_printout=True, plot_bar=False,
                 plot_radius=.95, plot_text_size=20, plot_cmap='viridis', plot_width=50):

        """Initialise the SOM network.

        Args:
            netHeight (int): Number of nodes along the first dimension.
            netWidth (int): Numer of nodes along the second dimension.
            data (np.array or list): N-dimensional dataset.
            loadFile (str, optional): Name of file to load containing information 
                to initialise the network weights.
            PCI (boolean): Activate/Deactivate Principal Component Analysis to set
                the initial value of weights
            PBC (boolean): Activate/Deactivate periodic boundary conditions,
                warning: only quality threshold clustering algorithm works with PBC.
            n_jobs (int) [WORK IN PROGRESS]: Number of parallel processes (-1 use all available)   
        """
        self.set_plot_settings(dpi=plot_dpi, path=plot_path, show=plot_show, printout=plot_printout, 
                               bar=plot_bar, radius=plot_radius, text_size=plot_text_size, cmap= plot_cmap, 
                               width=plot_width)
    
        """ Switch to activate special workflow if running the colours example. """
        self.colorEx=False
        
        """ Switch to activate periodic PCA weights initialisation. """
        self.PCI=bool(PCI)

        """ Switch to activate periodic boundary conditions. """
        self.PBC=bool(PBC)

        """ Activate light parallelization. """
        #TODO:
        #self.n_jobs=n_jobs

        if self.PBC==True:
            print("Periodic Boundary Conditions active.")
        else:
            print("Periodic Boundary Conditions inactive.")

        self.nodeList=[]
        self.data=data.reshape(np.array([data.shape[0], data.shape[1]]))
        self.clusters= []

        """ Load the weights from file, generate them randomly or from PCA. """

        if loadFile==None:
            self.netHeight = netHeight
            self.netWidth = netWidth

            minVal,maxVal=[],[]
            pcaVec=[]

            if self.PCI==True:
                print("The weights will be initialised with PCA.")
            
                pca = PCA(n_components = 2)
                pca.fit(self.data)
                pcaVec=pca.components_
            
            else:
                print("The weights will be initialised randomly.")

                for i in range(self.data.shape[1]):
                    minVal.append(np.min(self.data[:,i]))
                    maxVal.append(np.max(self.data[:,i]))
            
            for x in range(self.netWidth):
                for y in range(self.netHeight):
                    self.nodeList.append(somNode(x,y, self.data.shape[1], self.netHeight, self.netWidth, self.PBC,\
                        minVal=minVal, maxVal=maxVal, pcaVec=pcaVec))

        else:   
            print('The weights will be loaded from file.')

            if loadFile.endswith('.npy')==False:
                loadFile=loadFile+'.npy'
            weiArray=np.load(loadFile)
            #add something to check that data and array have the same dimensions,
            #or that they are mutually exclusive
            self.netHeight = int(weiArray[0][0])
            self.netWidth = int(weiArray[0][1])
            self.PBC= bool(weiArray[0][2])

            #start from 1 because 0 contains info on the size of the network
            countWei=1
            for x in range(self.netWidth):
                for y in range(self.netHeight):
                    self.nodeList.append(somNode(x,y, self.data.shape[1], self.netHeight, self.netWidth, self.PBC, weiArray=weiArray[countWei]))
                    countWei+=1
                    
    def set_plot_settings(self, dpi=70, path='./', show=True, printout=True, bar=False,
                                radius= 0.95, text_size=20, cmap='viridis', width=50):
        self.dpi= dpi
        self.path= path
        self.show= show
        self.printout= printout
        self.bar= bar
        self.radius= radius
        self.text_size= text_size * 50/width
        self.cmap= cmap
        self.widthP= width
        
    def get_custom_topographic_error(self, data):
        '''
            Returns a meassure of the topographic error.
            It is NOT the actual topographic error, but something similar
        '''
        error= 0
        for row in data:
            bmu1, bmu2 = self.find_bmu(row)
            error+= bmu1.get_distance(bmu2.weights)
        return error/len(data)

    def save(self, fileName='somNet_trained', path='./'):
    
        """Saves the network dimensions, the pbc and nodes weights to a file.

        Args:
            fileName (str, optional): Name of file where the data will be saved.
            
        """
        
        
        weiArray=[np.zeros(len(self.nodeList[0].weights))]
        weiArray[0][0],weiArray[0][1],weiArray[0][2]=self.netHeight, self.netWidth, int(self.PBC)
        for node in self.nodeList:
            weiArray.append(node.weights)
        np.save(os.path.join(path,fileName), np.asarray(weiArray))
    

    def update_sigma(self, iter):
    
        """Update the gaussian sigma.

        Args:           
            iter (int): Iteration number.
            
        """
    
        self.sigma = self.startSigma * np.exp(-iter/self.tau);
    

    def update_lrate(self, iter):
    
        """Update the learning rate.

        Args:           
            iter (int): Iteration number.
            
        """
        
        self.lrate =  self.startLearnRate * np.exp(-iter/self.epochs);
    

    def find_bmu(self, vec):
    
        """Find the best matching unit (BMU) for a given vector.

        Args:           
            vec (np.array): The vector to match.
            
        Returns:            
            bmu (somNode): The best matching unit node.
            
        """
        bmu1= self.nodeList[0]
        minVal=np.finfo(np.float).max
        for node in self.nodeList:
            dist=node.get_distance(vec)
            if dist < minVal:
                minVal=dist
                bmu2= bmu1
                bmu1= node
        return bmu1, bmu2
            

    def train(self, startLearnRate=0.01, epochs=-1):
    
        """Train the SOM.

        Args:
            startLearnRate (float): Initial learning rate.
            epochs (int): Number of training iterations. If not selected (or -1)
                automatically set epochs as 10 times the number of datapoints
            
        """
        
        print("Training SOM... 0%", end=' ')
        self.startSigma = max(self.netHeight, self.netWidth)/2
        self.startLearnRate = startLearnRate
        if epochs==-1:
            epochs=self.data.shape[0]*10
        self.epochs=epochs
        self.tau = self.epochs/np.log(self.startSigma)
    
        #TODO:
        #Parallel(n_jobs=self.n_jobs)(delayed(my_func)(c, K, N) for c in inputs)

        for i in range(self.epochs):

            if i%100==0:
                print(("\rTraining SOM... "+str(int(i*100.0/self.epochs))+"%" ), end=' ')

            self.update_sigma(i)
            self.update_lrate(i)
            
            """ Train with the bootstrap-like method: 
                instead of using all the training points, a random datapoint is chosen with substitution
                for each iteration and used to update the weights of all the nodes.
            """
            
            inputVec = self.data[np.random.randint(0, self.data.shape[0]), :].reshape(np.array([self.data.shape[1]]))
            
            bmu, _=self.find_bmu(inputVec)
            
            for node in self.nodeList:
                node.update_weights(inputVec, self.sigma, self.lrate, bmu)

        print("\rTraining SOM... done!")

        
    def nodes_graph(self, colnum=0, colname=None, colors=[], return_ax_only=False):
    
        """Plot a 2D map with hexagonal nodes and weights values

        Args:
            colnum (int): The index of the weight that will be shown as colormap.
            show (bool, optional): Choose to display the plot.
            printout (bool, optional): Choose to save the plot to a file.
            colname (str, optional): Name of the column to be shown on the map.
        """

        if not colname:
            colname = str(colnum)

        centers = [[node.pos[0],node.pos[1]] for node in self.nodeList]
        fig= self.get_fig()

        if self.colorEx==True:
            cols = [[np.float(node.weights[0]),np.float(node.weights[1]),np.float(node.weights[2])]for node in self.nodeList]   
            ax = hx.plot_hex(fig, centers, cols, radius=self.radius, cmap=self.cmap)
            ax.set_title('Node Grid with Color Features', size=self.text_size)
            printName=os.path.join(self.path,'nodesColors.png')

        else:
            cols = [node.weights[colnum] for node in self.nodeList]
            ax = hx.plot_hex(fig, centers, cols, edges=colors, radius=self.radius)
            if colname is None:
                ax.set_title('Feature #' +  str(colnum), size=self.text_size)
            else:
                ax.set_title(str(colname), size=self.text_size)
                
            if self.bar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.0)
                cbar=plt.colorbar(ax.collections[0], cax=cax)
                cbar.set_label(colname, size=self.text_size, labelpad=10)
                cbar.ax.tick_params(labelsize=self.text_size)
            plt.sca(ax)
            printName=os.path.join(self.path,'SOM_'+str(colnum if colname is None else colname)+'.png')
            
        if not return_ax_only:
            if self.printout==True:
                plt.savefig(printName, bbox_inches='tight', dpi=self.dpi)
            if self.show==True:
                plt.show()
            if self.show!=False and self.printout!=False:
                plt.clf()
            
        return ax

    def get_diffs(self):
        '''
            Obtain summed distance to neighbouring nodes
        '''
        neighbours=[]
        for node in self.nodeList:
            nodelist=[]
            for nodet in self.nodeList:
                if node != nodet and node.get_nodeDistance(nodet) <= 1.001:
                    nodelist.append(nodet)
            neighbours.append(nodelist)     
            
        diffs = []
        for node, neighbours in zip(self.nodeList, neighbours):
            diff=0
            for nb in neighbours:
                dist= node.get_distance(nb.weights)
                diff=diff + dist
            diffs.append(diff)  

        centers = [[node.pos[0],node.pos[1]] for node in self.nodeList]
        
        return diffs, neighbours, centers
    
    def get_fig(self):
        
        xInch = self.netWidth*self.widthP/self.dpi 
        yInch = self.netHeight*self.widthP/self.dpi
        
        return plt.figure(figsize=(xInch, yInch), dpi=self.dpi)

    def diff_graph(self, colors=[], return_ax_only=False):
    
        """Plot a 2D map with nodes and weights difference among neighbouring nodes.

        Args:
            show (bool, optional): Choose to display the plot.
            printout (bool, optional): Choose to save the plot to a file.
            returns (bool, optional): Choose to return the difference value.

        Returns:
            (list): difference value for each node.             
        """
        
        diffs, neighbours, centers= self.get_diffs()
        fig= self.get_fig()

        ax = hx.plot_hex(fig, centers, diffs, edges=colors, radius=self.radius, cmap=self.cmap)
        ax.set_title('Weights Difference', size=self.text_size)
            
        if self.bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.0)
            cbar=plt.colorbar(ax.collections[0], cax=cax)
            cbar.ax.tick_params(labelsize=self.text_size)
        plt.sca(ax)

        printName=os.path.join(self.path,'SOM_difference.png')
          
        if not return_ax_only:
            if self.printout==True:
                plt.savefig(printName, bbox_inches='tight', dpi=self.dpi)
            if self.show==True:
                plt.show()
            if self.show!=False and self.printout!=False:
                plt.clf()
                
        return ax

    def project(self, array, colnum=-1, colname= None, 
                plot_mode='number', show_clusters=False, return_only=False):

        """Project the datapoints of a given array to the 2D space of the 
            SOM by calculating the bmus. If requested plot a 2D map with as 
            implemented in nodes_graph and adds circles to the bmu
            of each datapoint in a given array.

        Args:
            array (np.array): An array containing datapoints to be mapped.
            colnum (int): The index of the weight that will be shown as colormap. 
                If not chosen, the difference map will be used instead.
            show (bool, optional): Choose to display the plot.
            printout (bool, optional): Choose to save the plot to a file.
            colname (str, optional): Name of the column to be shown on the map.
            plot_mode (str, optional): How to plot the number of datapoints: as 'number' or as 'scatter'
            
        Returns:
            (list): bmu x,y position for each input array datapoint. 
            
        """            
        if not colname:
            colname = str(colnum)

        hash_point= lambda p: round(p[0], 2)*1000 + round(p[1], 2)  
        bmuList, cls, points_per_node=[], [], {}
        for i in range(array.shape[0]):
            point= self.find_bmu(array[i,:])[0].pos
            hashed_point= hash_point(point)
            bmuList.append(point)
            
            if hashed_point in points_per_node:
                points_per_node[hashed_point][1]+= 1
            else:
                points_per_node[hashed_point]= [np.array(point), 1]
                
            if self.colorEx==True:
                cls.append(array[i,:])
            else: 
                if colnum==-1:
                    cls.append('#ffffff')
                else: 
                    cls.append(array[i,colnum])

        if (self.show==True or self.printout==True) and not return_only:
            if show_clusters:
                raise NotImplementedError('TO DO: Show clusters around cells as edges')
                #_, colors= self.get_clusters()
            else:
                colors= []
            
            #Call nodes_graph/diff_graph to first build the 2D map of the nodes
            if self.colorEx==True:
                printName=os.path.join(self.path,'colorProjection.png')
                self.nodes_graph(colnum, False, False)
                plt.scatter([pos[0] for pos in bmuList],[pos[1] for pos in bmuList], color=cls,  
                        s=500, edgecolor='#ffffff', linewidth=5, zorder=10)
                plt.title('Datapoints Projection', size=20)
                
            else:   
                if colnum==-1:
                    printName=os.path.join(self.path,'SOM_projection_difference.png')
                    ax= self.diff_graph(return_ax_only=True, colors=colors)
                    ax.set_title('Datapoints Projection on Weights Difference', size=20)
                else:
                    printName=os.path.join(self.path,'SOM_projection_'+str(colnum if colname is None else colname)+'.png')
                    ax= self.nodes_graph(colnum, colname=colname, return_ax_only=True, colors=colors)
                    
                if plot_mode=='scatter':
                    #A random perturbation is added to the points positions so that data 
                    #belonging plotted to the same bmu will be visible in the plot  
                    plt.scatter([pos[0]-0.125+np.random.rand()*0.25 for pos in bmuList],
                                [pos[1]-0.125+np.random.rand()*0.25 for pos in bmuList], c=cls, cmap=cm.viridis,
                                s=400, linewidth=0, zorder=10)
                elif plot_mode=='number':
                    for _,(point, repetitions) in points_per_node.items():
                        plt.annotate(repetitions, color='w',
                                     xy=point-0.15*min(1, repetitions*10/30), size=min(30/100*self.widthP, repetitions*10))
                else:
                    pass
            
            if self.printout==True:
                plt.savefig(printName, bbox_inches='tight', dpi=self.dpi)
            if self.show==True:
                plt.show()
            plt.clf()
        
        #Return the x,y coordinates of bmus, useful for the clustering function.
        return [[pos[0],pos[1]] for pos in bmuList] 
        
        
    def find_clusters(self, array, type='qthresh', cutoff=5, quant=0.2, percent=0.02, numcl=8,\
                savefile=True, filetype='dat'):
    
        """Clusters the data in a given array according to the SOM trained map.
            The clusters can also be plotted.

        Args:
            array (np.array): An array containing datapoints to be clustered.
            type (str, optional): The type of clustering to be applied, so far only quality threshold (qthresh) 
                and density peak algorithma are directly implemented, other algorithms require sklearn, and do not
                support PBC
            cutoff (float, optional): Cutoff for the quality threshold algorithm. This also doubles as
                maximum distance of two points to be considered in the same cluster with DBSCAN.
            percent (float, optional): The percentile that defines the reference distance in density peak clustering (dpeak).
            numcl (int, optional): The number of clusters for K-Means clustering
            quant (float, optional): Quantile used to calculate the bandwidth of the mean shift algorithm.
            savefile (bool, optional): Choose to save the resulting clusters in a text file.
            filetype (string, optional): Format of the file where the clusters will be saved (csv or dat)
            show (bool, optional): Choose to display the plot.
            printout (bool, optional): Choose to save the plot to a file.
            
        Returns:
            (list of int): A nested list containing the clusters with indexes of the input array points.
            
        """

        """ Call project to first find the bmu for each array datapoint, but without producing any graph. """

        bmuList= self.project(array, return_only=True)
        clusters=[]

        if type=='qthresh':
            #Cluster according to the quality threshold algorithm (slow!).
            clusters = qt.qualityThreshold(bmuList, cutoff, self.PBC, self.netHeight, self.netWidth)

        elif type=='dpeak':
            #Cluster according to the density peak algorithm.
            clusters = dp.densityPeak(bmuList, PBC=self.PBC, netHeight=self.netHeight, netWidth=self.netWidth)

        elif type in ['MeanShift', 'DBSCAN', 'KMeans']:
            #Cluster according to algorithms implemented in sklearn.
            if self.PBC==True:
                print("Warning: Only Quality Threshold and Density Peak clustering work with PBC")
            try:
        
                if type=='MeanShift':
                    bandwidth = cluster.estimate_bandwidth(np.asarray(bmuList), quantile=quant, n_samples=500)
                    cl = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(bmuList)
                
                if type=='DBSCAN':
                    cl = cluster.DBSCAN(eps=cutoff, min_samples=5).fit(bmuList)     
                
                if type=='KMeans':
                    cl= cluster.KMeans(n_clusters=numcl).fit(bmuList)

                clLabs = cl.labels_                 
                    
                for i in np.unique(clLabs):
                    clList=[]
                    tmpList=range(len(bmuList))
                    for j,k in zip(tmpList,clLabs):
                        if i==k:
                            clList.append(j)
                    clusters.append(clList)     
            except:
                print(('Unexpected error: ', sys.exc_info()[0]))
                raise
        else:
            sys.exit("Error: unkown clustering algorithm " + type)
        
        if savefile==True:
            file=open(os.path.join(self.path,'SOM_' + type + '_clusters.' + filetype), 'w')
            if filetype=='csv':
                separator=','
            else: 
                separator=' '
            for line in clusters:
                for id in line: file.write(str(id)+separator)
                file.write('\n')
            file.close()
                
        xc, yc, color=[],[],[]
        for i in range(len(clusters)):
            for c in clusters[i]:
                #again, invert y and x to be consistent with the previous maps
                xc.append(bmuList[int(c)][0])
                yc.append(self.netHeight-bmuList[int(c)][1])  
                color.append(i/len(clusters))
        
        if self.printout==True or self.show==True:
            printName=os.path.join(self.path,'SOM_' + type + '_clusters.png')
                        
            fig= self.get_fig()
            #ax.scatter(xc, yc, color=color, label='Cluster %d'%(i))
            ax = hx.plot_hex(fig, list(zip(xc, yc)), color, cmap='Set3', radius=self.radius)
            ax.set_title('Clusters', size=self.text_size)
            plt.sca(ax)

            plt.gca().invert_yaxis()
            ax.axis('off')
                
            if self.printout==True:
                plt.savefig(printName, bbox_inches='tight', dpi=self.dpi)
            if self.show==True:
                plt.show()
            plt.clf()   
            
        self.clusters, self.xc, self.yc, self.color = clusters, xc, yc, color
        
    def get_clusters(self):
        if self.clusters==[]:
            raise ValueError('Clusters not yet computed, please run find_clusters() before')
        return self.clusters, self.color
    
    def show_exogenous(self, array, exogenous, colname=None):
    
        """Plots an exogenous variable into the grid

        Args:
            array (np.array): An array containing datapoints to be clustered.
            array (np.array): An array containing the value of the variable to be plotted
            
        Returns:
            ax: The matplotlib.pyplot ax
            
        """

        hash_point= lambda p: round(p[0], 2)*1000 + round(p[1], 2)  
        mean_per_node= {} # hash_point: (point_coords, sum, N) 
        
        for row, ex in zip(array, exogenous):
            point= self.find_bmu(row)[0].pos
            hashed_point= hash_point(point)
            
            if hashed_point in mean_per_node:
                mean_per_node[hashed_point][2]+= 1
                mean_per_node[hashed_point][1]+= ex
            else:
                mean_per_node[hashed_point]= [np.array(point), ex, 1]
                
        points= [point for _,(point, s, N) in mean_per_node.items()]
        values= [s/N for _,(point, s, N) in mean_per_node.items()]
                    
        printName= os.path.join(self.path, 'SOM_%s.png'%('exogenous' if colname is None else colname))
                        
        fig= self.get_fig()
        cmap= 'Set3' if len(np.unique(exogenous)) < 8 else self.cmap #Use a set cmap if the exog. var. is a class.
        ax = hx.plot_hex(fig, points, values, cmap=cmap, radius=self.radius)
        ax.set_title('Exogenous variable' if colname is None else colname, size=self.text_size)
        
        if self.bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.0)
            cbar=plt.colorbar(ax.collections[0], cax=cax)
            cbar.ax.tick_params(labelsize=self.text_size)
        
        plt.sca(ax)

        plt.gca().invert_yaxis()
        ax.axis('off')
                
        if self.printout==True:
            plt.savefig(printName, bbox_inches='tight', dpi=self.dpi)
        if self.show==True:
            plt.show()
        plt.clf()   
        
class somNode:
    
    """ Single Kohonen SOM Node class. """
    
    def __init__(self, x, y, numWeights, netHeight, netWidth, PBC, minVal=[], maxVal=[], pcaVec=[], weiArray=[]):
    
        """Initialise the SOM node.

        Args:
            x (int): Position along the first network dimension.
            y (int): Position along the second network dimension
            numWeights (int): Length of the weights vector.
            netHeight (int): Network height, needed for periodic boundary conditions (PBC)
            netWidth (int): Network width, needed for periodic boundary conditions (PBC)
            PBC (bool): Activate/deactivate periodic boundary conditions.
            minVal(np.array, optional): minimum values for the weights found in the data
            maxVal(np.array, optional): maximum values for the weights found in the data
            pcaVec(np.array, optional): Array containing the two PCA vectors.
            weiArray (np.array, optional): Array containing the weights to give
                to the node if a file was loaded.

                
        """
    
        self.PBC=PBC
        self.pos = hx.coorToHex(x,y)
        self.weights = []

        self.netHeight=netHeight
        self.netWidth=netWidth

        if weiArray==[] and pcaVec==[]:
            #select randomly in the space spanned by the data
            for i in range(numWeights):
                self.weights.append(np.random.random()*(maxVal[i]-minVal[i])+minVal[i])
        elif weiArray==[] and pcaVec!=[]:
            #select uniformly in the space spanned by the PCA vectors
            self.weights= (x-self.netWidth/2)*2.0/self.netWidth * pcaVec[0] + (y-self.netHeight/2)*2.0/self.netHeight *pcaVec[1]
        else:
            for i in range(numWeights):
                self.weights.append(weiArray[i])

    
    def get_distance(self, vec):
    
        """Calculate the distance between the weights vector of the node and a given vector.

        Args:
            vec (np.array): The vector from which the distance is calculated.
            
        Returns: 
            (float): The distance between the two weight vectors.
        """
    
        sum=0
        if len(self.weights)==len(vec):
            for i in range(len(vec)):
                sum+=(self.weights[i]-vec[i])*(self.weights[i]-vec[i])
            return np.sqrt(sum)
        else:
            sys.exit("Error: dimension of nodes != input data dimension!")

    def get_nodeDistance(self, node):
    
        """Calculate the distance within the network between the node and another node.

        Args:
            node (somNode): The node from which the distance is calculated.
            
        Returns:
            (float): The distance between the two nodes.
            
        """

        if self.PBC==True:

            """ Hexagonal Periodic Boundary Conditions """
            
            if self.netHeight%2==0:
                offset=0
            else: 
                offset=0.5

            return  np.min([np.sqrt((self.pos[0]-node.pos[0])*(self.pos[0]-node.pos[0])\
                                +(self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1])),
                            #right
                            np.sqrt((self.pos[0]-node.pos[0]+self.netWidth)*(self.pos[0]-node.pos[0]+self.netWidth)\
                                +(self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1])),
                            #bottom 
                            np.sqrt((self.pos[0]-node.pos[0]+offset)*(self.pos[0]-node.pos[0]+offset)\
                                +(self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)),
                            #left
                            np.sqrt((self.pos[0]-node.pos[0]-self.netWidth)*(self.pos[0]-node.pos[0]-self.netWidth)\
                                +(self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1])),
                            #top 
                            np.sqrt((self.pos[0]-node.pos[0]-offset)*(self.pos[0]-node.pos[0]-offset)\
                                +(self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4)),
                            #bottom right
                            np.sqrt((self.pos[0]-node.pos[0]+self.netWidth+offset)*(self.pos[0]-node.pos[0]+self.netWidth+offset)\
                                +(self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)),
                            #bottom left
                            np.sqrt((self.pos[0]-node.pos[0]-self.netWidth+offset)*(self.pos[0]-node.pos[0]-self.netWidth+offset)\
                                +(self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)),
                            #top right
                            np.sqrt((self.pos[0]-node.pos[0]+self.netWidth-offset)*(self.pos[0]-node.pos[0]+self.netWidth-offset)\
                                +(self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4)),
                            #top left
                            np.sqrt((self.pos[0]-node.pos[0]-self.netWidth-offset)*(self.pos[0]-node.pos[0]-self.netWidth-offset)\
                                +(self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4))])
                        
        else:
            return np.sqrt((self.pos[0]-node.pos[0])*(self.pos[0]-node.pos[0])\
                +(self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1]))



    def update_weights(self, inputVec, sigma, lrate, bmu):
    
        """Update the node Weights.

        Args:
            inputVec (np.array): A weights vector whose distance drives the direction of the update.
            sigma (float): The updated gaussian sigma.
            lrate (float): The updated learning rate.
            bmu (somNode): The best matching unit.
        """
    
        dist=self.get_nodeDistance(bmu)
        gauss=np.exp(-dist*dist/(2*sigma*sigma))

        #if gauss>0: #pointless
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - gauss*lrate*(self.weights[i]-inputVec[i])
        
def run_colorsExample(path='./'):   

    """Example of usage of SimpSOM: a number of vectors of length three
        (corresponding to the RGB values of a color) are used to briefly train a small network.
        Different example graphs are then printed from the trained network.     
    """ 

    """Try to create the folder"""
    if path!='./':
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
                
    raw_data =np.asarray([[1, 0, 0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0.2,0.2,0.5]])
    labels=['red','green','blue','yellow','magenta','cyan','indigo']

    print("Welcome to SimpSOM (Simple Self Organizing Maps) v1.3.4!\nHere is a quick example of what this library can do.\n")
    print("The algorithm will now try to map the following colors: ", end=' ')
    for i in range(len(labels)-1):
            print((labels[i] + ", "), end=' ') 
    print("and " + labels[-1]+ ".\n")
    
    net = somNet(20, 20, raw_data, PBC=True) 
    
    net.colorEx=True
    net.train(0.01, 10000)

    print("Saving weights and a few graphs...", end=' ')
    net.save('colorExample_weights', path=path)
    net.nodes_graph(path=path)
    
    net.diff_graph(path=path)
    net.project(raw_data, labels=labels, path=path)
    net.cluster(raw_data, type='qthresh', path=path) 
    
    print("done!")


if __name__ == "__main__":

    run_colorsExample()
