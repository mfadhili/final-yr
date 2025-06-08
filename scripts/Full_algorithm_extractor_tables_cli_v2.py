#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mppt_cli.py

Run all 18 MPPT algorithms with custom population size and iteration count.
Generates detailed per‐algorithm CSVs, a summary metrics CSV, and a power‐vs‐iteration CSV.
"""
import argparse
import numpy as np
import pandas as pd
from math import gamma, sin, pi
from scipy.stats import qmc, wilcoxon

# -------------------------------------------------------------------------------
# 1️⃣ Initial Seed for Reproducibility
# -------------------------------------------------------------------------------
np.random.seed(49)

# -------------------------------------------------------------------------------
# 2️⃣ CLI Argument Parsing
# -------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 18 MPPT algorithms with custom pop_size and max_iter"
    )
    parser.add_argument(
        "-p", "--pop_size", type=int, default=30,
        help="Population size for all algorithms (default: 30)"
    )
    parser.add_argument(
        "-m", "--max_iter", type=int, default=100,
        help="Max number of iterations (default: 100)"
    )
    return parser.parse_args()

# -------------------------------------------------------------------------------
# 3️⃣ Global Parameters (overridden in main)
# -------------------------------------------------------------------------------
pop_size           = None
max_iter           = None
initial_irradiance = 800   # [W/m²]
T_value            = 25    # [°C]

# these will be set = pop_size in main()
num_particles    = 30
num_food_sources = 30
num_locusts      = 30
hm_size          = 30
n_nests          = 30

# SA params
initial_temp_sa = 50
cooling_rate    = 0.95

# HS params
HMCR = 0.9
PAR  = 0.3
bw   = 5

# CSA params
clone_factor     = 5
mutation_rate    = 0.2
replacement_rate = 0.2

# CS params
pa       = 0.25
alpha_cs = 0.01
beta_cs  = 1.5

# ABO params
p_abo = 0.8
c_abo = 0.01
a_exp = 0.1

# DE params
F  = 0.8
CR = 0.9

# -------------------------------------------------------------------------------
# 4️⃣ Helper: Export detailed CSV
# -------------------------------------------------------------------------------
def export_csv_results(records, filename):
    pd.DataFrame(records).to_csv(filename, index=False)
    print(f"Saved detailed CSV: {filename}")

# -------------------------------------------------------------------------------
# 5️⃣ PV System Model & Objective
# -------------------------------------------------------------------------------
def pv_system_model(V, G, T):
    I_sc           = 10
    V_oc           = 100
    Temp_coeff_V   = -0.005
    T_ref          = 25
    Max_efficiency = 0.85
    V_oc_adj       = V_oc*(1 + Temp_coeff_V*(T - T_ref))
    I              = I_sc*(1 - np.exp(-V/V_oc_adj))*(G/1000)
    P              = V*I
    P_max          = Max_efficiency*V_oc_adj*I_sc
    return P if P <= P_max else -np.inf

def objective_function(params, G, T):
    V = params[0]
    if V < 0 or V > 100:
        return -np.inf
    return pv_system_model(V, G, T)

# -------------------------------------------------------------------------------
# 6️⃣ Levy flight for Cuckoo Search
# -------------------------------------------------------------------------------
def levy_flight(beta):
    sigma = (gamma(1+beta)*sin(pi*beta/2)/
             (gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u, v = np.random.randn(), np.random.randn()
    return u / (abs(v)**(1/beta))

# -------------------------------------------------------------------------------
# 7️⃣ 18 MPPT Algorithm Definitions (all identical to your CLI skeleton + Sobol/Halton)
# -------------------------------------------------------------------------------

def HippopotamusAlgorithm(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    population = np.random.uniform(0,100,(pop_size,1))
    fitness    = np.array([objective_function(ind, G, T) for ind in population])
    best_sol   = population[np.argmax(fitness)].copy()
    best_fit   = np.max(fitness)
    conv, Vh, Gh, Th = [], [], [], []
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        for i in range(pop_size):
            partner = population[np.random.choice([j for j in range(pop_size) if j!=i])]
            if np.random.rand()<0.5:
                new = population[i] + np.random.uniform(-1,1)*(partner-population[i])
            else:
                new = population[i] + 0.5*(best_sol-population[i])
            fit_new = objective_function(new, G, T)
            if fit_new>fitness[i]:
                population[i]=new; fitness[i]=fit_new
        best_idx = np.argmax(fitness)
        best_sol = population[best_idx].copy()
        best_fit = fitness[best_idx]
        conv.append(best_fit); Vh.append(best_sol[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": best_fit,
                "Best Voltage": float(best_sol[0]),
                "Fitness Array": fitness.tolist(),
                "Population Array": population.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "hippopotamus_detailed_results.csv")
    return best_sol, best_fit, conv, Vh, Gh, Th

def SobolHippopotamusAlgorithm(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    sampler = qmc.Sobol(d=1, scramble=True)
    sample  = sampler.random_base2(m=int(np.ceil(np.log2(pop_size))))
    population = qmc.scale(sample[:pop_size],[0],[100]).reshape(pop_size,1)
    fitness    = np.array([objective_function(ind, G, T) for ind in population])
    best_sol   = population[np.argmax(fitness)].copy()
    best_fit   = np.max(fitness)
    conv, Vh, Gh, Th = [], [], [], []
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        for i in range(pop_size):
            partner = population[np.random.choice([j for j in range(pop_size) if j!=i])]
            if np.random.rand()<0.5:
                new = population[i] + np.random.uniform(-1,1)*(partner-population[i])
            else:
                new = population[i] + 0.5*(best_sol-population[i])
            fit_new = objective_function(new, G, T)
            if fit_new>fitness[i]:
                population[i]=new; fitness[i]=fit_new
        best_idx = np.argmax(fitness)
        best_sol = population[best_idx].copy()
        best_fit = fitness[best_idx]
        conv.append(best_fit); Vh.append(best_sol[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": best_fit,
                "Best Voltage": float(best_sol[0]),
                "Fitness Array": fitness.tolist(),
                "Population Array": population.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "sobol_hippo_detailed_results.csv")
    return best_sol, best_fit, conv, Vh, Gh, Th

def HaltonHippopotamusAlgorithm(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    sampler = qmc.Halton(d=1, scramble=True)
    sample  = sampler.random(n=pop_size)
    population = qmc.scale(sample,[0],[100]).reshape(pop_size,1)
    fitness    = np.array([objective_function(ind, G, T) for ind in population])
    best_sol   = population[np.argmax(fitness)].copy()
    best_fit   = np.max(fitness)
    conv, Vh, Gh, Th = [], [], [], []
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        for i in range(pop_size):
            partner = population[np.random.choice([j for j in range(pop_size) if j!=i])]
            if np.random.rand()<0.5:
                new = population[i] + np.random.uniform(-1,1)*(partner-population[i])
            else:
                new = population[i] + 0.5*(best_sol-population[i])
            fit_new = objective_function(new, G, T)
            if fit_new>fitness[i]:
                population[i]=new; fitness[i]=fit_new
        best_idx = np.argmax(fitness)
        best_sol = population[best_idx].copy()
        best_fit = fitness[best_idx]
        conv.append(best_fit); Vh.append(best_sol[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": best_fit,
                "Best Voltage": float(best_sol[0]),
                "Fitness Array": fitness.tolist(),
                "Population Array": population.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "halton_hippo_detailed_results.csv")
    return best_sol, best_fit, conv, Vh, Gh, Th

def TLBO(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    population = np.random.uniform(0,100,(pop_size,1))
    fitness    = np.array([objective_function(ind, G, T) for ind in population])
    best_sol   = population[np.argmax(fitness)].copy()
    best_fit   = np.max(fitness)
    conv, Vh, Gh, Th = [], [], [], []
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        teacher = population[np.argmax(fitness)]
        mean    = np.mean(population,axis=0)
        TF      = np.random.randint(1,3)
        for i in range(pop_size):
            new = population[i] + np.random.rand()*(teacher - TF*mean)
            fit_new = objective_function(new, G, T)
            if fit_new>fitness[i]:
                population[i]=new; fitness[i]=fit_new
        best_idx = np.argmax(fitness)
        best_sol = population[best_idx].copy()
        best_fit = fitness[best_idx]
        conv.append(best_fit); Vh.append(best_sol[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": best_fit,
                "Best Voltage": float(best_sol[0]),
                "Fitness Array": fitness.tolist(),
                "Population Array": population.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "tlbo_detailed_results.csv")
    return best_sol, best_fit, conv, Vh, Gh, Th

def GeneticAlgorithm(pop_size, max_iter, G, T,
                     mutation_rate=0.1, crossover_rate=0.8,
                     export_csv=False):
    data_records = [] if export_csv else None
    population = np.random.uniform(0,100,(pop_size,1))
    fitness    = np.array([objective_function(ind, G, T) for ind in population])
    best_sol   = population[np.argmax(fitness)].copy()
    best_fit   = np.max(fitness)
    conv, Vh   = [], []
    Gh = [G]*max_iter; Th = [T]*max_iter
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        idx = np.argsort(fitness)[::-1]
        population = population[idx]; fitness = fitness[idx]
        new_pop = []
        while len(new_pop)<pop_size:
            if np.random.rand()<crossover_rate:
                p1,p2 = population[np.random.choice(pop_size,2,replace=False)]
                cp     = np.random.rand()
                c1 = cp*p1 + (1-cp)*p2
                c2 = (1-cp)*p1 + cp*p2
                new_pop.extend([c1,c2])
            else:
                new_pop.append(population[np.random.randint(pop_size)])
        new_pop = np.array(new_pop)[:pop_size]
        mask    = np.random.rand(*new_pop.shape)<mutation_rate
        mut_vals= np.random.uniform(-2,2,new_pop.shape)*mask
        new_pop+=mut_vals
        new_fit = np.array([objective_function(ind, G, T) for ind in new_pop])
        population, fitness = new_pop, new_fit
        best_idx = np.argmax(fitness)
        best_sol = population[best_idx].copy()
        best_fit = fitness[best_idx]
        conv.append(best_fit); Vh.append(best_sol[0])
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": best_fit,
                "Best Voltage": float(best_sol[0]),
                "Fitness Array": fitness.tolist(),
                "Population Array": population.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "genetic_detailed_results.csv")
    return best_sol, best_fit, conv, Vh, Gh, Th

def PSO_MPPT(num_particles, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    w, c1, c2 = 0.7,1.5,1.5
    pos = np.random.uniform(0,100,(num_particles,1))
    vel = np.random.uniform(-10,10,(num_particles,1))
    pbest_pos = pos.copy()
    pbest_fit = np.array([objective_function([float(x)],G,T) for x in pos])
    g_idx = np.argmax(pbest_fit)
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = pbest_fit[g_idx]
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        for i in range(num_particles):
            r1,r2 = np.random.rand(),np.random.rand()
            vel[i]=(w*vel[i]+c1*r1*(pbest_pos[i]-pos[i])+
                    c2*r2*(gbest_pos-pos[i]))
            pos[i]=np.clip(pos[i]+vel[i],0,100)
            f = objective_function(pos[i],G,T)
            if f>pbest_fit[i]:
                pbest_fit[i]=f; pbest_pos[i]=pos[i].copy()
        idx=np.argmax(pbest_fit)
        if pbest_fit[idx]>gbest_fit:
            gbest_fit=pbest_fit[idx]; gbest_pos=pbest_pos[idx].copy()
        conv.append(gbest_fit); Vh.append(gbest_pos[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": gbest_fit,
                "Best Voltage": float(gbest_pos[0]),
                "Fitness Array": pbest_fit.tolist(),
                "Population Array": pos.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "pso_detailed_results.csv")
    return gbest_pos, gbest_fit, conv, Vh, Gh, Th

def ABC_MPPT(num_food_sources, max_iter, G, T, limit=20, export_csv=False):
    data_records = [] if export_csv else None
    foods = np.random.uniform(0,100,(num_food_sources,1))
    fit   = np.array([objective_function(src,G,T) for src in foods])
    trial = np.zeros(num_food_sources)
    best_idx = np.argmax(fit)
    gbest_pos = foods[best_idx].copy()
    gbest_fit = fit[best_idx]
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        for i in range(num_food_sources):
            k = np.random.choice([j for j in range(num_food_sources) if j!=i])
            phi = np.random.uniform(-1,1)
            new = foods[i] + phi*(foods[i]-foods[k])
            new = np.clip(new,0,100)
            fnew = objective_function(new,G,T)
            if fnew>fit[i]:
                foods[i]=new; fit[i]=fnew; trial[i]=0
            else:
                trial[i]+=1
        if fit.sum()>0:
            prob=fit/fit.sum()
        else:
            prob=np.ones(num_food_sources)/num_food_sources
        for i in range(num_food_sources):
            if np.random.rand()<prob[i]:
                k = np.random.choice([j for j in range(num_food_sources) if j!=i])
                phi=np.random.uniform(-1,1)
                new=foods[i]+phi*(foods[i]-foods[k]); new=np.clip(new,0,100)
                fnew=objective_function(new,G,T)
                if fnew>fit[i]:
                    foods[i]=new; fit[i]=fnew; trial[i]=0
                else:
                    trial[i]+=1
        for i in range(num_food_sources):
            if trial[i]>limit:
                foods[i]=np.random.uniform(0,100,(1,1))
                fit[i]=objective_function(foods[i],G,T)
                trial[i]=0
        idx=np.argmax(fit)
        if fit[idx]>gbest_fit:
            gbest_fit=fit[idx]; gbest_pos=foods[idx].copy()
        G = G*(0.9+0.2*np.random.rand())
        conv.append(gbest_fit); Vh.append(gbest_pos[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": gbest_fit,
                "Best Voltage": float(gbest_pos[0]),
                "Fitness Array": fit.tolist(),
                "Population Array": foods.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "abc_detailed_results.csv")
    return gbest_pos, gbest_fit, conv, Vh, Gh, Th

def SA_MPPT(max_iter, G, T, initial_temp_sa=50, cooling_rate=0.95, export_csv=False):
    data_records = [] if export_csv else None
    current = np.array([np.random.uniform(0,100)])
    fcur    = objective_function(current,G,T)
    best    = current.copy(); fbest = fcur
    temp    = initial_temp_sa
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        delta = np.random.uniform(-5,5)
        cand = np.clip(current+delta,0,100)
        fc  = objective_function(cand,G,T)
        df  = fc - fcur
        if df>=0 or np.random.rand()<np.exp(df/temp):
            current, fcur = cand, fc
        if fcur>fbest:
            best, fbest = current.copy(), fcur
        conv.append(fbest); Vh.append(best[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": fbest,
                "Best Voltage": float(best[0]),
                "Fitness Array": [float(fcur)],
                "Population Array": [float(current[0])]
            })
        temp *= cooling_rate
    if export_csv:
        export_csv_results(data_records, "sa_detailed_results.csv")
    return best, fbest, conv, Vh, Gh, Th

def GWO_MPPT(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    pos = np.random.uniform(0,100,(pop_size,1))
    fit = np.array([objective_function(p,G,T) for p in pos])
    idxs = np.argsort(fit)[::-1]
    a_pos, a_fit = pos[idxs[0]].copy(), fit[idxs[0]]
    b_pos = pos[idxs[1]].copy() if pop_size>1 else a_pos
    d_pos = pos[idxs[2]].copy() if pop_size>2 else b_pos
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        a_val = 2 - it*(2/max_iter)
        for i in range(pop_size):
            for (wp,cp) in [(a_pos,1),(b_pos,2),(d_pos,3)]:
                r1,r2 = np.random.rand(),np.random.rand()
                A,C   = 2*a_val*r1-a_val, 2*r2
                D     = abs(C*wp - pos[i])
                X     = wp - A*D
                pos[i] = (pos[i] + X)/2
            pos[i] = np.clip(pos[i],0,100)
        G = G*(0.9+0.2*np.random.rand())
        fit = np.array([objective_function(p,G,T) for p in pos])
        idxs = np.argsort(fit)[::-1]
        a_pos, a_fit = pos[idxs[0]].copy(), fit[idxs[0]]
        if pop_size>1: b_pos = pos[idxs[1]].copy()
        if pop_size>2: d_pos = pos[idxs[2]].copy()
        conv.append(a_fit); Vh.append(a_pos[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": float(a_fit),
                "Best Voltage": float(a_pos[0]),
                "Fitness Array": fit.tolist(),
                "Population Array": pos.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "gwo_detailed_results.csv")
    return a_pos, a_fit, conv, Vh, Gh, Th

def HarmonySearchMPPT(hm_size, max_iter, G, T,
                      HMCR=0.9, PAR=0.3, bw=5,
                      export_csv=False):
    data_records = [] if export_csv else None
    hm = np.random.uniform(0,100,(hm_size,1))
    fm = np.array([objective_function([x],G,T) for x in hm.flatten()])
    idx = np.argsort(fm)[::-1]
    hm, fm = hm[idx], fm[idx]
    best, bf = hm[0].copy(), fm[0]
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        nh = np.zeros((1,))
        if np.random.rand()<HMCR:
            nh[0] = np.random.choice(hm.flatten())
            if np.random.rand()<PAR:
                nh[0]+=np.random.uniform(-bw,bw)
        else:
            nh[0]=np.random.uniform(0,100)
        nh[0]=np.clip(nh[0],0,100)
        fh = objective_function(nh,G,T)
        wi = np.argmin(fm)
        if fh>fm[wi]:
            hm[wi], fm[wi] = nh.copy(), fh
            idx = np.argsort(fm)[::-1]
            hm, fm = hm[idx], fm[idx]
        if fm[0]>bf:
            best, bf = hm[0].copy(), fm[0]
        conv.append(bf); Vh.append(best[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": float(bf),
                "Best Voltage": float(best[0]),
                "Fitness Array": fm.tolist(),
                "Population Array": hm.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "hs_detailed_results.csv")
    return best, bf, conv, Vh, Gh, Th

def ClonalSelectionMPPT(pop_size, max_iter, G, T,
                        clone_factor=5, mutation_rate=0.2, replacement_rate=0.2,
                        export_csv=False):
    data_records = [] if export_csv else None
    pop = np.random.uniform(0,100,(pop_size,1))
    fit = np.array([objective_function([x],G,T) for x in pop.flatten()])
    bi  = np.argmax(fit)
    best, bf = pop[bi].copy(), fit[bi]
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        idx = np.argsort(fit)[::-1]
        pop, fit = pop[idx], fit[idx]
        clones=[]
        for rank,cand in enumerate(pop):
            n_cl = int(np.ceil(clone_factor*(pop_size-rank)/pop_size))
            clones += [cand.copy() for _ in range(n_cl)]
        clones = np.array(clones)
        mut_cl = clones + np.random.uniform(-mutation_rate*100,mutation_rate*100,clones.shape)
        mut_cl = np.clip(mut_cl,0,100)
        cf = np.array([objective_function([x],G,T) for x in mut_cl.flatten()])
        comb_pop = np.vstack((pop,mut_cl))
        comb_fit = np.concatenate((fit,cf))
        bi2 = np.argsort(comb_fit)[::-1][:pop_size]
        pop, fit = comb_pop[bi2], comb_fit[bi2]
        nr = int(np.ceil(replacement_rate*pop_size))
        if nr>0:
            new = np.random.uniform(0,100,(nr,1))
            nf  = np.array([objective_function([x],G,T) for x in new.flatten()])
            wi  = np.argsort(fit)[:nr]
            pop[wi], fit[wi] = new, nf
        bi = np.argmax(fit)
        if fit[bi]>bf:
            best, bf = pop[bi].copy(), fit[bi]
        conv.append(bf); Vh.append(best[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": float(bf),
                "Best Voltage": float(best[0]),
                "Fitness Array": fit.tolist(),
                "Population Array": pop.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "csa_detailed_results.csv")
    return best, bf, conv, Vh, Gh, Th

def LocustSwarmMPPT(num_locusts, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    pos = np.random.uniform(0,100,(num_locusts,1))
    fit = np.array([objective_function([x],G,T) for x in pos.flatten()])
    bi  = np.argmax(fit)
    best, bf = pos[bi].copy(), fit[bi]
    conv, Vh, Gh, Th = [],[],[],[]
    alpha,beta,gamma_ = 0.5,0.3,0.2
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        mean_p = np.mean(pos)
        for i in range(num_locusts):
            r1,r2,r3 = np.random.rand(),np.random.rand(),np.random.rand()
            new = pos[i] + alpha*r1*(best-pos[i]) + beta*r2*(mean_p-pos[i]) + gamma_*r3*100*(np.random.rand()-0.5)
            pos[i]=np.clip(new,0,100)
        fit = np.array([objective_function([x],G,T) for x in pos.flatten()])
        bi  = np.argmax(fit)
        if fit[bi]>bf:
            best, bf = pos[bi].copy(), fit[bi]
        conv.append(bf); Vh.append(best[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": float(bf),
                "Best Voltage": float(best[0]),
                "Fitness Array": fit.tolist(),
                "Population Array": pos.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "lsa_detailed_results.csv")
    return best, bf, conv, Vh, Gh, Th

def EmperorPenguinOptimizer(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    pos = np.random.uniform(0,100,(pop_size,1))
    fit = np.array([objective_function([x],G,T) for x in pos.flatten()])
    bi  = np.argmax(fit)
    best, bf = pos[bi].copy(), fit[bi]
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        a = 2 - it*(2/max_iter)
        mean_p = np.mean(pos)
        for i in range(pop_size):
            r1,r2,r3 = np.random.rand(),np.random.rand(),np.random.rand()
            new = (pos[i]
                   + a*r1*(best-pos[i])
                   + 0.5*r2*(mean_p-pos[i])
                   + 0.1*r3*100*(np.random.rand()-0.5))
            pos[i]=np.clip(new,0,100)
        fit = np.array([objective_function([x],G,T) for x in pos.flatten()])
        bi  = np.argmax(fit)
        if fit[bi]>bf:
            best, bf = pos[bi].copy(), fit[bi]
        conv.append(bf); Vh.append(best[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": float(bf),
                "Best Voltage": float(best[0]),
                "Fitness Array": fit.tolist(),
                "Population Array": pos.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "epo_detailed_results.csv")
    return best, bf, conv, Vh, Gh, Th

def CuckooSearchMPPT(n_nests, max_iter, G, T,
                     pa=0.25, alpha_cs=0.01, beta_cs=1.5,
                     export_csv=False):
    data_records = [] if export_csv else None
    nests = np.random.uniform(0,100,(n_nests,1))
    fit   = np.array([objective_function([x],G,T) for x in nests.flatten()])
    bi    = np.argmax(fit)
    best, bf = nests[bi].copy(), fit[bi]
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        new_nests = np.empty_like(nests)
        for i in range(n_nests):
            step = levy_flight(beta_cs)
            new_nests[i]=np.clip(nests[i]+alpha_cs*step,0,100)
        fnew = np.array([objective_function([x],G,T) for x in new_nests.flatten()])
        for i in range(n_nests):
            j = np.random.randint(n_nests)
            if fnew[i]>fit[j]:
                nests[j], fit[j] = new_nests[i].copy(), fnew[i]
        num_ab = int(pa*n_nests)
        worst = np.argsort(fit)[:num_ab]
        nests[worst] = np.random.uniform(0,100,(num_ab,1))
        fit[worst]   = np.array([objective_function([x],G,T) for x in nests[worst].flatten()])
        bi = np.argmax(fit)
        if fit[bi]>bf:
            best, bf = nests[bi].copy(), fit[bi]
        conv.append(bf); Vh.append(best[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": float(bf),
                "Best Voltage": float(best[0]),
                "Fitness Array": fit.tolist(),
                "Population Array": nests.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "cuckoo_detailed_results.csv")
    return best, bf, conv, Vh, Gh, Th

def ABO_MPPT(pop_size, max_iter, G, T,
             p=0.8, c=0.01, a_exp=0.1,
             export_csv=False):
    data_records = [] if export_csv else None
    pos = np.random.uniform(0,100,(pop_size,1))
    fit = np.array([objective_function([x],G,T) for x in pos.flatten()])
    bi  = np.argmax(fit)
    best, bf = pos[bi].copy(), fit[bi]
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        for i in range(pop_size):
            fi = c*(fit[i]**a_exp) if fit[i]!=-np.inf else 0
            r  = np.random.rand()
            if r<p:
                new = pos[i] + r*(best-pos[i])*fi
            else:
                a,b = np.random.choice(pop_size,2,replace=False)
                new = pos[i] + r*(pos[a]-pos[b])*fi
            pos[i] = np.clip(new,0,100)
        fit = np.array([objective_function([x],G,T) for x in pos.flatten()])
        bi  = np.argmax(fit)
        if fit[bi]>bf:
            best, bf = pos[bi].copy(), fit[bi]
        conv.append(bf); Vh.append(best[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": float(bf),
                "Best Voltage": float(best[0]),
                "Fitness Array": fit.tolist(),
                "Population Array": pos.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "abo_detailed_results.csv")
    return best, bf, conv, Vh, Gh, Th

def HEM_MPPT(pop_size, max_iter, G, T,
             selection_rate=0.3, mutation_rate=0.1,
             crossover_rate=0.7, diversity_rate=0.1,
             export_csv=False):
    data_records = [] if export_csv else None
    pop = np.random.uniform(0,100,(pop_size,1))
    fit = np.array([objective_function([x],G,T) for x in pop.flatten()])
    bi  = np.argmax(fit)
    best, bf = pop[bi].copy(), fit[bi]
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        ec = int(np.ceil(selection_rate*pop_size))
        idx=np.argsort(fit)[::-1]
        elites = pop[idx[:ec]]
        children=[]
        while len(children)<pop_size-ec:
            if np.random.rand()<crossover_rate:
                p1,p2 = elites[np.random.choice(ec,2,replace=False)]
                w      = np.random.rand()
                child  = w*p1 + (1-w)*p2
            else:
                child = elites[np.random.randint(ec)]
            child += np.random.uniform(-mutation_rate*100,mutation_rate*100,child.shape)
            child = np.clip(child,0,100)
            children.append(child)
        children = np.array(children)
        r        = np.random.rand(*children.shape)
        children = children + r*(best-children)
        children = np.clip(children,0,100)
        dc = int(np.ceil(diversity_rate*pop_size))
        ri = np.random.uniform(0,100,(dc,1))
        new_pop = np.vstack((elites,children))
        if new_pop.shape[0]>pop_size:
            new_pop=new_pop[:pop_size]
        else:
            new_pop[-dc:]=ri
        pop = new_pop
        fit = np.array([objective_function([x],G,T) for x in pop.flatten()])
        bi  = np.argmax(fit)
        if fit[bi]>bf:
            best, bf = pop[bi].copy(), fit[bi]
        conv.append(bf); Vh.append(best[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": float(bf),
                "Best Voltage": float(best[0]),
                "Fitness Array": fit.tolist(),
                "Population Array": pop.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "hem_detailed_results.csv")
    return best, bf, conv, Vh, Gh, Th

def WhiteSharkOptimizerMPPT(pop_size, max_iter, G, T, export_csv=False):
    data_records = [] if export_csv else None
    pos = np.random.uniform(0,100,(pop_size,1))
    fit = np.array([objective_function([x],G,T) for x in pos.flatten()])
    bi  = np.argmax(fit)
    best, bf = pos[bi].copy(), fit[bi]
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        hf = np.exp(-it/max_iter)
        for i in range(pop_size):
            r1,r2 = np.random.rand(),np.random.rand()
            new = (pos[i] + r1*hf*(best-pos[i])
                   + r2*(np.random.rand()-0.5)*10)
            pos[i]=np.clip(new,0,100)
        fit = np.array([objective_function([x],G,T) for x in pos.flatten()])
        bi  = np.argmax(fit)
        if fit[bi]>bf:
            best, bf = pos[bi].copy(), fit[bi]
        conv.append(bf); Vh.append(best[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": float(bf),
                "Best Voltage": float(best[0]),
                "Fitness Array": fit.tolist(),
                "Population Array": pos.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "wso_detailed_results.csv")
    return best, bf, conv, Vh, Gh, Th

def DifferentialEvolutionMPPT(pop_size, max_iter, G, T,
                             F=0.8, CR=0.9, export_csv=False):
    data_records = [] if export_csv else None
    pop = np.random.uniform(0,100,(pop_size,1))
    fit = np.array([objective_function([x],G,T) for x in pop.flatten()])
    bi  = np.argmax(fit)
    best, bf = pop[bi].copy(), fit[bi]
    conv, Vh, Gh, Th = [],[],[],[]
    for it in range(max_iter):
        G = G*(0.9+0.2*np.random.rand())
        new_pop = np.empty_like(pop)
        for i in range(pop_size):
            idx = list(range(pop_size)); idx.remove(i)
            r = np.random.choice(idx,3,replace=False)
            r1,r2,r3 = pop[r[0]], pop[r[1]], pop[r[2]]
            mutant = r1 + F*(r2-r3)
            trial  = mutant if np.random.rand()<CR else pop[i]
            new_pop[i]=np.clip(trial,0,100)
        new_fit = np.array([objective_function([x],G,T) for x in new_pop.flatten()])
        for i in range(pop_size):
            if new_fit[i]>fit[i]:
                pop[i], fit[i] = new_pop[i].copy(), new_fit[i]
        bi = np.argmax(fit)
        if fit[bi]>bf:
            best, bf = pop[bi].copy(), fit[bi]
        conv.append(bf); Vh.append(best[0]); Gh.append(G); Th.append(T)
        if export_csv:
            data_records.append({
                "Iteration": it+1,
                "Best Power Output": float(bf),
                "Best Voltage": float(best[0]),
                "Fitness Array": fit.tolist(),
                "Population Array": pop.flatten().tolist()
            })
    if export_csv:
        export_csv_results(data_records, "de_detailed_results.csv")
    return best, bf, conv, Vh, Gh, Th

# -------------------------------------------------------------------------------
# 8️⃣ Main Execution
# -------------------------------------------------------------------------------
def main():
    args = parse_args()
    global pop_size, max_iter
    global num_particles, num_food_sources, num_locusts, hm_size, n_nests

    pop_size       = args.pop_size
    max_iter       = args.max_iter
    num_particles  = num_food_sources = num_locusts = hm_size = n_nests = pop_size

    # re-seed to reproduce identical sequences
    np.random.seed(49)

    # list of (name, function, params...)
    algos = [
        ("Hippopotamus",       HippopotamusAlgorithm,        (pop_size, max_iter, initial_irradiance, T_value, True)),
        ("Sobol-Hippopotamus", SobolHippopotamusAlgorithm,    (pop_size, max_iter, initial_irradiance, T_value, True)),
        ("Halton-Hippopotamus",HaltonHippopotamusAlgorithm,   (pop_size, max_iter, initial_irradiance, T_value, True)),
        ("TLBO",               TLBO,                         (pop_size, max_iter, initial_irradiance, T_value, True)),
        ("Genetic Alg",        GeneticAlgorithm,             (pop_size, max_iter, initial_irradiance, T_value, 0.1, 0.8, True)),
        ("PSO",                PSO_MPPT,                     (num_particles, max_iter, initial_irradiance, T_value, True)),
        ("ABC",                ABC_MPPT,                     (num_food_sources, max_iter, initial_irradiance, T_value, 20, True)),
        ("SA",                 SA_MPPT,                      (max_iter, initial_irradiance, T_value, initial_temp_sa, cooling_rate, True)),
        ("GWO",                GWO_MPPT,                     (pop_size, max_iter, initial_irradiance, T_value, True)),
        ("HS",                 HarmonySearchMPPT,            (hm_size, max_iter, initial_irradiance, T_value, HMCR, PAR, bw, True)),
        ("CSA",                ClonalSelectionMPPT,          (pop_size, max_iter, initial_irradiance, T_value, clone_factor, mutation_rate, replacement_rate, True)),
        ("LSA",                LocustSwarmMPPT,              (num_locusts, max_iter, initial_irradiance, T_value, True)),
        ("EPO",                EmperorPenguinOptimizer,      (pop_size, max_iter, initial_irradiance, T_value, True)),
        ("Cuckoo Search",      CuckooSearchMPPT,             (n_nests, max_iter, initial_irradiance, T_value, pa, alpha_cs, beta_cs, True)),
        ("ABO",                ABO_MPPT,                     (pop_size, max_iter, initial_irradiance, T_value, p_abo, c_abo, a_exp, True)),
        ("HEM",                HEM_MPPT,                     (pop_size, max_iter, initial_irradiance, T_value, 0.3, 0.1, 0.7, 0.1, True)),
        ("WSO",                WhiteSharkOptimizerMPPT,      (pop_size, max_iter, initial_irradiance, T_value, True)),
        ("DE",                 DifferentialEvolutionMPPT,    (pop_size, max_iter, initial_irradiance, T_value, F, CR, True)),
    ]

    results = []
    for name, func, params in algos:
        best_sol, best_fit, conv, Vh, Gh, Th = func(*params)
        results.append((name, conv, Vh))

    # Summary metrics CSV
    summary = []
    for name, conv, Vh in results:
        bp = max(conv)
        ir = conv.index(bp)+1
        bv = Vh[ir-1]
        summary.append({
            "Algorithm": name,
            "Best Voltage (V)": bv,
            "Best Power (W)": bp,
            "Iteration Reached": ir
        })
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("summary_metrics.csv", index=False)
    print("Saved summary_metrics.csv")

    # Power vs Iteration CSV
    table = {"Iteration": list(range(1, max_iter+1))}
    for name, conv, _ in results:
        table[name] = conv
    df_power = pd.DataFrame(table)
    df_power.to_csv("power_vs_iteration.csv", index=False)
    print("Saved power_vs_iteration.csv")

if __name__ == "__main__":
    main()
