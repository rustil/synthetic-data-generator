import math
import random
from array import array
import pickle
# --- LCIO dependencies ---
from pyLCIO import UTIL, EVENT, IMPL, IO, IOIMPL
import numpy as np

#---- number of events  -----
nevt = 5

outfile = "wgan.slcio"

#--------------------------------------------

wrt = IOIMPL.LCFactory.getInstance().createLCWriter( )

wrt.open( outfile , EVENT.LCIO.WRITE_NEW ) 

print " opened outfile: " , outfile

random.seed()

wgan_file = np.load("cache_wgan.npz", allow_pickle=True)
wgan = wgan_file['x']


#========== MC particle properties ===================
p = 50.
genstat  = 1
charge = 0.
mass = 0.00 
decayLen = 1.e32 
pdg = 22


# write a RunHeader
run = IMPL.LCRunHeaderImpl() 
run.setRunNumber( 0 ) 
run.parameters().setValue("Generator","WGAN")
run.parameters().setValue("PDG", pdg )
wrt.writeRunHeader( run ) 


for j in range( 0, nevt ):

    

    ### MC particle Collections
    colmc = IMPL.LCCollectionVec( EVENT.LCIO.MCPARTICLE ) 

    energy   = math.sqrt( mass*mass  + p*p )

    ## we are shooting 90 deg. ECAL 
    px = 0.00 
    py = p 
    pz = 0.00 

    vx = 0.00
    vy = 50.00
    vz = 0.000

    epx = 0.00
    epy = 1800
    epz = 0.00

    momentum  = array('f',[ px, py, pz ] )  
    vertex = array('d',[vx,vy,vz])
    endpoint = array('d', [epx,epy,epz])

    mcp = IMPL.MCParticleImpl() 
    mcp.setGeneratorStatus( genstat ) 
    mcp.setMass( mass )
    mcp.setPDG( pdg ) 
    mcp.setMomentum( momentum )
    mcp.setCharge( charge )
    mcp.setVertex(vertex)
    mcp.setEndpoint(endpoint)

    colmc.addElement( mcp )
    
    evt = IMPL.LCEventImpl() 
    evt.setEventNumber( j ) 
    evt.addCollection( colmc , "MCParticle" )


    ### Calorimeter Collections
    col = IMPL.LCCollectionVec( EVENT.LCIO.SIMCALORIMETERHIT ) 
    flag =  IMPL.LCFlagImpl(0) 
    flag.setBit( EVENT.LCIO.CHBIT_LONG )
    flag.setBit( EVENT.LCIO.CHBIT_ID1 )

    col.setFlag( flag.getFlag() )

    col.parameters().setValue(EVENT.LCIO.CellIDEncoding, 'system:0:5,module:5:3,stave:8:4,tower:12:4,layer:16:6,wafer:22:6,slice:28:4,cellX:32:-16,cellY:48:-16')
    
    evt.addCollection( col , "ECalBarrelSiHits" )
   

    nshowers = len(wgan[j])

    ### get from pickle file, loop over in the event cells
    for i in range(0,nshowers):
        sch = IMPL.SimCalorimeterHitImpl()
        energy = wgan[j][i][3]
        x = wgan[j][i][0]
        y = wgan[j][i][1]
        z = wgan[j][i][2]
        id0 = wgan[j][i][4]
        id1 = wgan[j][i][5]
        position = array('f',[x,y,z])
        #print x,y,z,energy
        sch.setPosition(position)
        sch.setEnergy(energy)
        sch.setCellID0(int(id0))
        sch.setCellID1(int(id1))

        
        col.addElement( sch )


    wrt.writeEvent( evt ) 


wrt.close() 
