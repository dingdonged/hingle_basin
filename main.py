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
#pw  is proppant weight
#pr is pump rate


production_file = "well production.csv" #this stores production data and well file names

def import_production_data():
    #we import production data from a file
    df = pd.read_csv(production_file)
    oil_total = []
    #sum up the oil production for each month and add it as a column for each well
    for ind,row in df.iterrows():
        total_oil_prod = 0
        for i in range(1,  12):
            total_oil_prod += row['oil '+str(i)]
        oil_total.append(total_oil_prod)
    df['Total Oil Production'] = oil_total
    
    width = []
    for well_name in df["well name"]:
        well_df = pd.read_csv(f"{well_name}.csv")
        #also take the max-min thickness and add that as a column for each well
        width.append(well_df['easting'][99]-well_df['easting'][0]) #max easting - min easting
    df['Well Width'] = width
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
    ax.scatter(x,y,z, s=1)
    plt.show()
    
def twod_cmap_scatter(x,y,z,zString):
    #plots x,y in 2d and uses z to vary color
    im = plt.scatter(x,y, s=1, c=z)
    cbar = plt.colorbar(im, orientation = 'vertical')
    cbar.set_label(zString, rotation=270, labelpad=20)
    plt.show()

def build_bundle(wells):
    #we build the map of the 'world' in terms of the quantities that define it
    #our goal will be to plot and smooth this
    #this will be called once so efficiency is not a priority
    #we also rename the columns functionally
    r = pd.DataFrame(columns=["x","y","phi","k","nu","E", "Cw", "Co", "t", "pw", "pr"])
    for d in wells.values():
        r = r.append(pd.DataFrame({'x': d["easting"], 'y': d["northing"], 'phi': d["porosity"], 'k': d["permeability"], 'nu': d["Poisson's ratio"], 
            'E': d["Young's Modulus"], 'Cw': d["water saturation"], 'Co': d["oil saturation"], 't': d["thickness (ft)"], 'pw': d["proppant weight (lbs)"], 'pr': d["pump rate (cubic feet/min)"]}))
    return r


def main():
    
    production_data = import_production_data()

    #now we log production data
    print("######### Production Data ##########")
    pprint(production_data.head())

    #we get the well logs by using the well names in production
    wells = import_well_data(production_data["well name"])
    total_oil = [] #list of total oil production for each well
    frack_lengths = [] #list of number of frack stages
    for well_name, data in wells.items():
        #print(f"{well_name}:")
        #pprint(data.head()) #We'll turn this diagnostic off soon. it clutters
        frack_lengths.append(data.shape[0]) #adds the num rows (or number of frack stages for one well)
        #print() #this is a simple diagnostic atm

    print()
    print("Frack stages: ")
    print(frack_lengths)

    print()
    print("Bundle data head:")
    bundle = build_bundle(wells)
    pprint(bundle.head())

    #we now graph x,y vs porosity for visualisation to locate trends in the data
    print("Plotting porosity")
    scatter_surface(bundle["x"], bundle["y"], bundle["phi"])
    twod_cmap_scatter(bundle["x"], bundle["y"], bundle["phi"], "Porosity")
    #for instance phi seems to be increasing with x and decreasing with y. the data looks roughly linear.
    print("Plotting permiability")
    scatter_surface(bundle["x"], bundle["y"], bundle["k"])
    twod_cmap_scatter(bundle["x"], bundle["y"], bundle["k"], "Permiability")
    print("Plotting oil saturation")
    scatter_surface(bundle["x"], bundle["y"], bundle["Co"])
    twod_cmap_scatter(bundle["x"], bundle["y"], bundle["Co"], "Oil Saturation")
    print("Plotting Poisson's ratio")
    scatter_surface(bundle["x"], bundle["y"], bundle["nu"])
    twod_cmap_scatter(bundle["x"], bundle["y"], bundle["nu"], "Poisson's Ratio")
    print("Plotting Young's Modulus")
    scatter_surface(bundle["x"], bundle["y"], bundle["E"])
    twod_cmap_scatter(bundle["x"], bundle["y"], bundle["E"], "Young's Modulus")
    print("Plotting Proppant Weight")
    scatter_surface(bundle["x"], bundle["y"], bundle["pw"])
    twod_cmap_scatter(bundle["x"], bundle["y"], bundle["pw"], "Proppant Weight")
    print("Plotting Pump Rate")
    scatter_surface(bundle["x"], bundle["y"], bundle["pr"])
    twod_cmap_scatter(bundle["x"], bundle["y"], bundle["pr"], "Pump Rate")
#     We don't need to plot water saturation AND oil saturation because they are just complements of each other 
#     print("Plotting water saturation")
#     scatter_surface(bundle["x"], bundle["y"], bundle["Cw"])
#     twod_cmap_scatter(bundle["x"], bundle["y"], bundle["Cw"], "Water Saturation")
#     print("Plotting thickness")
#     scatter_surface(bundle["x"], bundle["y"], bundle["t"])

    #plot total poil production vs well width
    im = plt.scatter(production_data['Well Width'], production_data["Total Oil Production"], s=10)
    plt.show()


if __name__ == "__main__":
    main()
