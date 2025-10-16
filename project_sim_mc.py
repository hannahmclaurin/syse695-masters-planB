import numpy as np                            
import matplotlib.pyplot as plt               
import pandas as pd                           

#mc descriptors (used https://www.datacamp.com/ tutorials on-monte-carlo-simulation-python)
N = 100000                                      #number of monte carlo iterations (high enough for stable statistics)
frameworks = ["NASA SE", "OpenSE", "Minimal"]  #the three se frameworks being compared
results = {fw: [] for fw in frameworks}        #dictionary to hold simulated science data loss for each framework

#sampling
def triangular_sample(low, mode, high, size):  #custom wrapper for triangular sampling
    return np.random.triangular(low, mode, high, size)

def bounded_normal(mean, sd, size, low=0, high=1):  #samples normal distribution clipped to 0 to 1 range
    vals = np.random.normal(mean, sd, size)
    return np.clip(vals, low, high)
#comments here are AI (GitHub CoPilot) generated based on provided inputs from analysis and its reasoning about how each framework's characteristics influence each factor

#mc simulation set-up
for fw in frameworks:                          #run simulation separately for each framework
    if fw=="NASA SE":
        D=bounded_normal(0.3,0.2,N)            #duration overrun: conservative mean, large variance (nasa's longer lifecycle)
        R=bounded_normal(0.5,0.02,N)           #rework: nasa emphasizes v&v, so higher expected effort on rework
        IRL=bounded_normal(7/9,0.05,N)         #integration readiness: strong systems focus, high expected irl
        Tm=bounded_normal(0.01,0.01,N)         #maintenance downtime: low due to high reliability engineering
        P=triangular_sample(0.1,0.12,0.20,N)   #design compromise: moderate due to rigorous stakeholder processes
        B1=triangular_sample(0.01,0.05,0.10,N) #budget reallocation: low-moderate due to process-driven change control
        I=triangular_sample(0.10,0.15,0.25,N)  #indirect impacts: moderate from procedural complexity
        Q=triangular_sample(0.05,0.2,0.35,N)   #measurement quality degradation: wider spread due to integration effects
        
    elif fw=="OpenSE":
        D=bounded_normal(0.3,0.3,N)            #duration overrun: broader spread due to looser phase controls
        R=bounded_normal(0.10,0.08,N)          #rework: lower than nasa due to lean iteration; more variability
        IRL=bounded_normal(6/9,0.08,N)         #integration readiness: moderate, with some risks due to less formality
        Tm=bounded_normal(0.05,0.02,N)         #maintenance downtime: expected moderate due to agile fixes
        P=triangular_sample(0.05,0.1,0.12,N)   #design compromise: moderate due to rapid prototyping
        B1=triangular_sample(0.03,0.10,0.20,N) #budget reallocation: higher than nasa due to emergent priorities
        I=triangular_sample(0.05,0.10,0.15,N)  #indirect impacts: moderate but managed through lean processes
        Q=triangular_sample(0.02,0.10,0.20,N)  #measurement quality degradation: narrower but can still be impactful

    else:  #minimal
        D=bounded_normal(0.3,0.5,N)            #duration overrun: high variability due to lack of planning
        R=bounded_normal(0.25,0.10,N)          #rework: high due to poor requirements traceability
        IRL=bounded_normal(3/9,0.12,N)         #integration readiness: low due to poor coordination and interface definition
        Tm=bounded_normal(0.15,0.05,N)         #maintenance downtime: high due to poor reliability practices
        P=triangular_sample(0.01,0.05,0.15,N)  #design compromise: variable depending on reactive decisions
        B1=triangular_sample(0.10,0.25,0.40,N) #budget reallocation: very high due to unplanned rework and scope shifts
        I=triangular_sample(0.0,0.01,0.05,N)   #indirect impacts: lower expected due to reduced procedural overhead
        Q=triangular_sample(0.01,0.03,0.07,N)  #measurement quality degradation: narrower but can accumulate

    #execute science data loss calculation
    L = (D + R + (1 - IRL) + Tm + P + B1 + I + Q) / 8  #equal-weighted average of 8 contributing factors
    L = np.clip(L, 0, 1)                               #ensure total loss is within 0 to 1 valid range
    results[fw] = L                                    #store simulation results for the framework

#results summary
summary = pd.DataFrame({
    fw: [np.mean(results[fw]), 
         np.std(results[fw]),                       
         np.percentile(results[fw], [5, 95])[0],    
         np.percentile(results[fw], [5, 95])[1],     
         np.mean(results[fw] > 0.1)]                  #probability of >10% science loss
    for fw in frameworks
}, index=["Mean L", "Std L", "P5", "P95", "P(L>10%)"])
print(summary.round(3))                               #round results for easier viewing

#plot
plt.figure(figsize=(8,5))                             #set plot size
for fw in frameworks:
    plt.hist(results[fw], bins=100, alpha=0.6, label=fw, density=True)  #density histogram for each framework
plt.xlabel("Science Data Loss (fraction of total yield)")               #x-axis label
plt.ylabel("Probability Density")                                       #y-axis label
plt.title("Monte Carlo Simulation of Science Data Loss by Framework")   #plot title
plt.legend()                                                            #show framework labels
plt.grid(alpha=0.3)                                                     #light grid for readability
plt.tight_layout()                                                      #prevent label overlap
plt.show()                                                              #display the plot