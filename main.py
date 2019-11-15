import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint

#noumenclature (todo: move to readme or seperate log):
#easting  = x
#northing = y
#the bundle refers to the properties of the subsurface at x,y (not the human chosen params)
#the following conventions are used in wikipedia:
#phi is porosity
#k is permiability
#nu is Poisson's Ratio
#E is Young's Modulous
#Cw is saturation of water
#Co is saturation of oil. Co+Cw=1 usually
#t is thickness


production_file = "well production.csv" #this stores production data and well file names

def import_production_data():
    #we import production data from a file
    df = pd.read_csv(production_file)
    return df #simple!

def import_well_data(well_names):
    #given a file name, import well log
    r = {} #zero initial return value
    for well_name in well_names:
        df = pd.read_csv(f"{well_name}.csv")
        r[well_name] = df

    return r

def scatter_surface(x,y,z):
    #3d plots a points above the xy plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, s=10)
    plt.show()

def build_bundle(wells):
    #we build the map of the 'world' in terms of the quantities that define it
    #our goal will be to plot and smooth this
    #this will be called once so efficiency is not a priority
    #we also rename the columns functionally
    r = pd.DataFrame(columns=["x","y","phi","k","nu","E", "Cw", "Co", "t"])
    for d in wells.values():
        r = r.append(pd.DataFrame({'x': d["easting"], 'y': d["northing"], 'phi': d["porosity"], 'k': d["permeability"], 'nu': d["Poisson's ratio"], 
            'E': d["Young's Modulus"], 'Cw': d["water saturation"], 'Co': d["oil saturation"], 't': d["thickness (ft)"]}))
    return r


def main():
    
    production_data = import_production_data()

    #now we log production data
    print("######### Production Data ##########")
    pprint(production_data.head())

    #we get the well logs by using the well names in production
    wells = import_well_data(production_data["well name"])
    frack_lengths = [] #list of number of frack stages
    for well_name, data in wells.items():
        print(f"{well_name}:")
        pprint(data.head()) #We'll turn this diagnostic off soon. it clutters
        frack_lengths.append(data.shape[0]) #adds the num rows (or number of frack stages for one well)
        print() #this is a simple diagnostic atm

    print()
    print("Frack stages: ")
    print(frack_lengths)

    print()
    print("Bundle data head:")
    bundle = build_bundle(wells)
    pprint(bundle.head())

    #we now graph x,y vs porosity for visualisation to locate trends in the data
    scatter_surface(bundle["x"], bundle["y"], bundle["phi"])
    #for instance phi seems to be increasing with x and decreasing with y. the data looks roughly linear.
    scatter_surface(bundle["x"], bundle["y"], bundle["k"])
    scatter_surface(bundle["x"], bundle["y"], bundle["Co"])






if __name__ == "__main__":
    main()



