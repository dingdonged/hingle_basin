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
    r = pd.DataFrame(columns=["x","y","phi","k","nu","E", "Cw", "Co", "t", "p", "pw", "pr", "oil", "frac",
                             "water", "ap", "rf", "fvf", "wl"])
    for d in wells.values():
        r = r.append(pd.DataFrame({'x': d["easting"], 'y': d["northing"], 'phi': d["porosity"], 'k': d["permeability"], 'nu': d["Poisson's ratio"], 
            'E': d["Young's Modulus"], 'Cw': d["water saturation"], 'Co': d["oil saturation"], 't': d["thickness (ft)"], 
            'p': d["pump rate (cubic feet/min)"], 'pw': d["proppant weight (lbs)"], 'pr': d["pump rate (cubic feet/min)"],
            'oil': d["oil"], 'frac': d["frac_stages"], 'water': d["water"], 'ap': d["average_pressure"],
            'rf': d["recovery_factor"], 'fvf': d["form_vol_factor"], 'wl': d["well_length"]}))
    return r


def main():
    
    production_data = import_production_data()

    #now we log production data
    print("######### Production Data ##########")
    pprint(production_data.head())

    #we get the well logs by using the well names in production
    wells = import_well_data(production_data["well name"])

#     for well_name, data in wells.items():
#         print(f"{well_name}:")
#         pprint(data.head()) #We'll turn this diagnostic off soon. it clutters
#         print() #this is a simple diagnostic atm
    frack_lengths = [] #list of number of frack stages
    cumulative_oil = production_data["oil 1"] #list of cumulative oil production
    cumulative_water = production_data["water 1"]
    for i in range(2,13):
        word = "oil " + str(i)
        water = "water " + str(i)
        cumulative_oil = cumulative_oil + production_data[word]
        cumulative_water = cumulative_water + production_data[water]

# compile all the data into wells
    i = 0
    for well_name, data in wells.items():
        well_length = data.easting.max()-data.easting.min()
        data["well_length"] = well_length
        frack_lengths.append(data.shape[0])
        data["oil"] = cumulative_oil[i]
        data["frac_stages"] = frack_lengths[i]
        data["water"] = cumulative_water[i]
        data["average_pressure"] = production_data["average pressure (Pa)"][i]
        data["recovery_factor"] = production_data["recovery factor"][i]
        data["form_vol_factor"] = production_data["formation volume factor"][i]
        i = i+1

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
#     We don't need to plot water saturation AND oil saturation because they are just complements of each other 
#     print("Plotting water saturation")
#     scatter_surface(bundle["x"], bundle["y"], bundle["Cw"])
#     twod_cmap_scatter(bundle["x"], bundle["y"], bundle["Cw"], "Water Saturation")
#     print("Plotting thickness")
#     scatter_surface(bundle["x"], bundle["y"], bundle["t"])

#     let's get the ml going
    bundle.isnull().any()
    bundle = bundle.fillna(method='ffill')
    Y = bundle["oil"].values
    X = bundle.drop(["oil", "Co"], axis=1).values
    plt.figure(figsize=(15,10))
    plt.tight_layout()
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
    
#     lasso regression
    lasso = Lasso()
    lasso.fit(X_train,y_train)
    train_score=lasso.score(X_train,y_train)
    test_score=lasso.score(X_test,y_test)
    coeff_used = np.sum(lasso.coef_!=0)
    print("training score:", train_score )
    print("test score: ", test_score)
    print("number of features used: ", coeff_used)
    
#     linear regression
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    regressor_train_score = regressor.score(X_train,y_train)
    regressor_test_score = regressor.score(X_test,y_test)
    print("train score:", regressor_train_score)
    print("test score:", regressor_test_score)





if __name__ == "__main__":
    main()
