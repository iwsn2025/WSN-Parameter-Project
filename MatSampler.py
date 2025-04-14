import numpy as np
import pandas as pd
from scipy.spatial import distance

class MatSampler:

    def __init__(self,simulator="tossim"):
        #set simulator
        self.simulator=simulator

    def readData(self):
        #load simulation and physical data

        self.sim = pd.read_csv("data/training/simulationData/%s.csv"%(self.simulator))
        self.phy = pd.read_csv("data/training/physicalData/physicalData.csv")

    def getMahalanobisDistanceInDifferentConfiguration(self):
        #get simulation physical distance in each network configurations within 5 shots physical data

        self.readData()

        #number of shots in simulation data
        n_sim_shots = 75
        #number of shots in physical data
        n_phy_shots = 5
        #number of configurations
        n_configs = 88

        mah_distances_all = []
        #for each network configuration
        for c in range(n_configs):
            #simulation physical distance in each network configuration
            mah_distances_i = []
            # for each shot
            for i in range(n_phy_shots):

                simulation_data = self.sim[self.sim["LA"]==c].loc[:, ~self.sim.columns.isin(['A', 'C','P','LA',"feature"])][0:n_sim_shots].values
                physical_sample = self.phy[self.phy["LA"]==c].loc[:, ~self.phy.columns.isin(['A', 'C','P','LA',"feature"])][i:i+1].values
                all_physical_sample = self.phy[self.phy["LA"]==c].loc[:, ~self.phy.columns.isin(['A', 'C','P','LA',"feature"])][0:5].values

                #calculate the centroid of simulation data
                centroid_sim = np.average(simulation_data,axis=0)

                #calculate the max distance between centoid and the simulation data points
                cov=np.cov(np.vstack([simulation_data,physical_sample]).T)

                #calculate the physical simulation gap
                phy_sim_dist = self.Mahalanobis(centroid_sim,physical_sample,cov=cov)
                mah_distances_i.append(phy_sim_dist)
            mah_distances_all.append({"simulation physical Mahalanobis distance in network configuration %d in 5 shots"%(c):mah_distances_i})


        print(mah_distances_all)
        return mah_distances_all

    def judgeByMahalanobis(self):
        #load physical and simulation Data
        self.readData()

        #number of shots in simulation data
        n_sim_shots = 50
        #number of shots in physical data
        n_phy_shots = 5
        #number of configurations
        n_configs = 88


        sample_count = np.ones(n_configs)*n_phy_shots


        #for each network configuration
        for c in range(n_configs):
            # for each shot
            for i in range(n_phy_shots):

                simulation_data = self.sim[self.sim["LA"]==c].loc[:, ~self.sim.columns.isin(['A', 'C','P','LA',"feature"])][0:n_sim_shots].values
                physical_sample = self.phy[self.phy["LA"]==c].loc[:, ~self.phy.columns.isin(['A', 'C','P','LA',"feature"])][i:i+1].values

                #calculate the centroid of simulation data
                centroid_sim = np.average(simulation_data,axis=0)

                #calculate the max distance between centoid and the simulation data points
                cov=np.cov(np.vstack([simulation_data,physical_sample]).T)
                l = [self.Mahalanobis(xi,centroid_sim,cov=cov) for xi in simulation_data]
                max_simulation_dis = np.max(l)

                #calculate the physical simulation gap
                phy_sim_dist = self.Mahalanobis(centroid_sim,physical_sample,cov=cov)

                if i<=1 :
                # if i<=1:
                    #mimimum physical data sample distance
                    #when physical sample number is less than two
                    #we use physical simulation distance as the previous physical distance
                     pre_i_phy = self.phy[self.phy["LA"]==c].loc[:, ~self.phy.columns.isin(['A', 'C','P','LA',"feature"])][0:1].values
                     physical_dists=[self.Mahalanobis(centroid_sim,pre_i_phy,cov=np.cov(np.vstack([simulation_data,pre_i_phy]).T))]

                elif i>1:
                     # get the previous physical data samples
                     pre_i_phy = self.phy[self.phy["LA"]==c].loc[:, ~self.phy.columns.isin(['A', 'C','P','LA',"feature"])][0:i].values

                     #calculate the physical distances between current sample and previous samples
                     physical_dists = [self.Mahalanobis(yi,physical_sample,cov=np.cov(np.vstack([pre_i_phy,physical_sample]).T))  for yi in pre_i_phy]


                #phy_phy_dist < phy_sim_dist or phy_sim_dist < sim_sim_dist
                if (min(physical_dists)< abs(phy_sim_dist-max_simulation_dis) or phy_sim_dist < max_simulation_dis )  and sample_count[c]==n_phy_shots :

                    # print("phy_phy_dist, phy_sim_dist, sim_sim_dist",max(physical_dists),phy_sim_dist,max_simulation_dis)
                    sample_count[c] = i+1
                    break

        return sample_count


    def sampleByMahalanobis(self,extraSample=0):

        #number of configurations
        n_configs = 88
        #get sample count in each network configuration
        sample_count = self.judgeByMahalanobis()

        #adding or removing extra sample
        print("sample count",sample_count)

        print(np.sum(sample_count))
        #concat all the samples in each configuration
        sample = []
        remaining = []
        for c in range(n_configs):
            item = pd.DataFrame(self.phy[self.phy["LA"]==c][0:int(sample_count[c])],columns=[ "A","C","P","B","L","R","LA"]).values
            sample.append(item)
            remaining.append(pd.DataFrame(self.phy[self.phy["LA"]==c][int(sample_count[c]):5],columns=[ "A","C","P","B","L","R","LA"]).values)

        sample = np.vstack(sample)
        remaining = np.vstack(remaining)


        if extraSample>0:
            sample = np.vstack([sample,remaining[0:extraSample]])
        else:
            sample = sample[0:len(sample)+extraSample]




        #save the sample to file
        pd.DataFrame(sample,columns=[ "A","C","P","B","L","R","LA"]).to_csv("data/training/physicalDataWithSamplingAlgorithm/%s_selected_samples.csv"%(self.simulator),index=False)

        #return the number of samples
        return len(sample)



    def Mahalanobis(self,y=None, data=None, cov=None):

        # calculateMahalanobis function to calculate
        # the Mahalanobis distance

        # print("cov",cov)

        if cov is  None:
            cov = np.cov(np.array(data).T)
        try:
            inv_covmat = np.linalg.inv(cov)
        except:
            inv_covmat = np.linalg.pinv(cov)


        mahal = distance.mahalanobis(y,data,inv_covmat)

        return mahal


def test():

    ss = ["ns3","cooja","tossim","omnet"]
    for i in ss:
        c = MatSampler(i).sampleByMahalanobis()
        print(i,"sample Count",c)

# test()
