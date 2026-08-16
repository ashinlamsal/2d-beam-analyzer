"""Direct-stiffness solver for horizontal 2D frame members."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Support:
    x: float
    kind: str

@dataclass(frozen=True)
class Load:
    kind: str
    magnitude: float = 0.0
    x: float = 0.0
    start: float = 0.0
    end: float = 0.0
    magnitude_end: float = 0.0

@dataclass
class AnalysisResult:
    nodes: np.ndarray
    displacements: np.ndarray
    reactions: np.ndarray
    element_forces: list[np.ndarray]
    element_loads: list[tuple[float, float]]

def _stiffness(E, I, A, L):
    k = np.zeros((6, 6)); ea, ei = E*A/L, E*I/L**3
    k[np.ix_([0,3],[0,3])] = ea*np.array([[1,-1],[-1,1]])
    ids=[1,2,4,5]
    k[np.ix_(ids,ids)] = ei*np.array([[12,6*L,-12,6*L],[6*L,4*L**2,-6*L,2*L**2],[-12,-6*L,12,-6*L],[6*L,2*L**2,-6*L,4*L**2]])
    return k

def _equiv(q0, q1, L):
    p=np.zeros(6); p[[1,2,4,5]]=[L*(7*q0+3*q1)/20,L**2*(3*q0+2*q1)/60,L*(3*q0+7*q1)/20,-L**2*(2*q0+3*q1)/60]
    return p

def _node(nodes, x):
    matches=np.flatnonzero(np.isclose(nodes,x,rtol=0,atol=1e-9))
    if len(matches)!=1: raise ValueError(f"Could not place a node at x = {x:g}.")
    return int(matches[0])

def solve_beam(length, E, I, A, supports, loads):
    if min(length,E,I,A)<=0: raise ValueError("Length, E, I, and A must be greater than zero.")
    if not supports: raise ValueError("Add supports before solving.")
    for s in supports:
        if s.kind not in {"pinned","roller","fixed"} or not 0<=s.x<=length: raise ValueError("Each support needs a valid type and position.")
    for load in loads:
        pos=[load.x] if load.kind in {"point","axial"} else [load.start,load.end]
        if load.kind not in {"point","axial","udl","uvl"} or any(x<0 or x>length for x in pos): raise ValueError("Each load must lie on the beam.")
        if load.kind in {"udl","uvl"} and load.end<=load.start: raise ValueError("Distributed load end must exceed its start.")
    positions=[0.,length]+[s.x for s in supports]
    for load in loads: positions += [load.x] if load.kind in {"point","axial"} else [load.start,load.end]
    nodes=np.array(sorted(set(round(x,10) for x in positions))); K=np.zeros((3*len(nodes),)*2); F=np.zeros(3*len(nodes)); qs=[]
    for e,(x1,x2) in enumerate(zip(nodes[:-1],nodes[1:])):
        L=x2-x1; ids=np.array([3*e,3*e+1,3*e+2,3*e+3,3*e+4,3*e+5]); K[np.ix_(ids,ids)]+=_stiffness(E,I,A,L); q0=q1=0.; mid=(x1+x2)/2
        for load in loads:
            if load.kind=="udl" and load.start<=mid<=load.end: q0+=load.magnitude; q1+=load.magnitude
            if load.kind=="uvl" and load.start<=mid<=load.end:
                span=load.end-load.start; q0+=load.magnitude+(load.magnitude_end-load.magnitude)*(x1-load.start)/span; q1+=load.magnitude+(load.magnitude_end-load.magnitude)*(x2-load.start)/span
        F[ids]+=_equiv(q0,q1,L); qs.append((q0,q1))
    for load in loads:
        if load.kind=="point": F[3*_node(nodes,load.x)+1]+=load.magnitude
        if load.kind=="axial": F[3*_node(nodes,load.x)]+=load.magnitude
    restrained=set()
    for s in supports:
        n=_node(nodes,s.x)
        if s.kind=="fixed": restrained.update([3*n,3*n+1,3*n+2])
        elif s.kind=="pinned": restrained.update([3*n,3*n+1])
        else: restrained.add(3*n+1)
    free=np.array([i for i in range(len(F)) if i not in restrained]); U=np.zeros(len(F))
    try: U[free]=np.linalg.solve(K[np.ix_(free,free)],F[free])
    except np.linalg.LinAlgError as exc: raise ValueError("Unstable structure: supports do not restrain rigid-body motion.") from exc
    forces=[]
    for e,(x1,x2) in enumerate(zip(nodes[:-1],nodes[1:])):
        ids=np.array([3*e,3*e+1,3*e+2,3*e+3,3*e+4,3*e+5]); forces.append(_stiffness(E,I,A,x2-x1)@U[ids]-_equiv(*qs[e],x2-x1))
    return AnalysisResult(nodes,U,K@U-F,forces,qs)

def sample_diagrams(result, points_per_element=50):
    xs=[]; axial=[]; shear=[]; moment=[]; deflection=[]
    for e,(x1,x2) in enumerate(zip(result.nodes[:-1],result.nodes[1:])):
        L=x2-x1; f=result.element_forces[e]; q0,q1=result.element_loads[e]; xi=np.linspace(0,L,points_per_element); r=xi/L; a=(q1-q0)/L
        V=f[1]+q0*xi+a*xi**2/2; M=-f[2]+f[1]*xi+q0*xi**2/2+a*xi**3/6; d=result.displacements[[3*e+1,3*e+2,3*e+4,3*e+5]]
        v=(1-3*r**2+2*r**3)*d[0]+L*(r-2*r**2+r**3)*d[1]+(3*r**2-2*r**3)*d[2]+L*(-r**2+r**3)*d[3]
        if e: xi,V,M,v=xi[1:],V[1:],M[1:],v[1:]
        xs.extend(x1+xi); shear.extend(V); moment.extend(M); deflection.extend(v); axial.extend(np.full(len(xi),-f[0]))
    return {"x":np.array(xs),"axial":np.array(axial),"shear":np.array(shear),"moment":np.array(moment),"deflection":np.array(deflection)}
