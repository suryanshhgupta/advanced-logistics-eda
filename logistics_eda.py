"""Week 3 logistics EDA: reproducible hypothetical dataset and charts.
Install: pip install pandas numpy matplotlib
Run: python week3_logistics_eda.py
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
OUT=Path("visualizations"); OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(42); n=600
regions=np.array(["North","South","East","West","Central"])
carriers=np.array(["Carrier A","Carrier B","Carrier C"]); modes=np.array(["Road","Rail","Air"])
region=rng.choice(regions,n,p=[.25,.20,.15,.22,.18]); carrier=rng.choice(carriers,n,p=[.4,.35,.25]); mode=rng.choice(modes,n,p=[.65,.22,.13])
volume=rng.integers(1,35,n); distance=np.clip(rng.normal(550,250,n),40,1500).round(1)
bt=distance/np.select([mode=="Air",mode=="Rail",mode=="Road"],[700,250,55])
re=pd.Series(region).map({"North":.4,"South":.7,"East":.9,"West":.2,"Central":.1}).to_numpy()
ce=pd.Series(carrier).map({"Carrier A":.2,"Carrier B":.8,"Carrier C":1.3}).to_numpy()
delivery=np.maximum(.3,bt+re+ce+rng.normal(1.8,1,n)); ix=rng.choice(n,15,False); delivery[ix]+=rng.uniform(3,8,len(ix))
cost=np.maximum(60,120+distance*.48+volume*5.5+np.select([mode=="Air",mode=="Rail",mode=="Road"],[320,70,0])+rng.normal(0,55,n))
delay=np.maximum(0,delivery-(distance/150+2.5))
df=pd.DataFrame({"shipment_id":[f"SHP{i:04}" for i in range(1,n+1)],"region":region,"carrier":carrier,"transport_mode":mode,"shipment_volume_units":volume,"distance_km":distance,"delivery_time_days":delivery.round(2),"transport_cost":cost.round(2),"delay_days":delay.round(2),"on_time_delivery":delay<=.5})
for col,k in [("transport_cost",5),("distance_km",3)]: df.loc[rng.choice(n,k,False),col]=np.nan
df.to_csv("hypothetical_logistics_dataset.csv",index=False)
print(df.describe(include="all")); print(df.isna().sum()); print(df[["shipment_volume_units","distance_km","delivery_time_days","transport_cost","delay_days"]].corr())
print(df.groupby("transport_mode")["transport_cost"].mean()); print(df.groupby("region")["delay_days"].mean())
plt.figure(figsize=(7,4)); plt.hist(df.delivery_time_days,bins=25,edgecolor="black"); plt.title("Distribution of Delivery Time"); plt.xlabel("Days"); plt.ylabel("Shipments"); plt.tight_layout(); plt.savefig(OUT/"01_delivery_time_distribution.png",dpi=150); plt.close()
plt.figure(figsize=(7,4)); df.groupby("transport_mode").transport_cost.mean().reindex(["Road","Rail","Air"]).plot(kind="bar",edgecolor="black"); plt.title("Average Transport Cost by Mode"); plt.tight_layout(); plt.savefig(OUT/"02_cost_by_mode.png",dpi=150); plt.close()
plt.figure(figsize=(7,4)); plt.scatter(df.distance_km,df.transport_cost,alpha=.65); plt.title("Distance vs Transport Cost"); plt.xlabel("Distance (km)"); plt.ylabel("Cost"); plt.tight_layout(); plt.savefig(OUT/"03_distance_vs_cost.png",dpi=150); plt.close()
plt.figure(figsize=(7,4)); df.groupby("region").delay_days.mean().reindex(["North","South","East","West","Central"]).plot(kind="bar",edgecolor="black"); plt.title("Average Delay by Region"); plt.tight_layout(); plt.savefig(OUT/"04_delay_by_region.png",dpi=150); plt.close()
num=["shipment_volume_units","distance_km","delivery_time_days","transport_cost","delay_days"]; corr=df[num].corr()
plt.figure(figsize=(7,5)); plt.imshow(corr,vmin=-1,vmax=1); plt.colorbar(); plt.xticks(range(len(num)),num,rotation=45,ha="right"); plt.yticks(range(len(num)),num)
for i in range(len(num)):
 for j in range(len(num)): plt.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8)
plt.tight_layout(); plt.savefig(OUT/"05_correlation_matrix.png",dpi=150); plt.close()
print("Saved CSV and charts.")
