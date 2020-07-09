# Running reconstruction with iLCsoft and ILDConfig 

After generation of `slcio` files of desired generative models, we would like to give them to reconstuction algoritm in `ILDConfig`. To do that, first check out the repo 

```
git clone --branch v02-01-pre02 https://github.com/iLCSoft/ILDConfig.git
```

We also need to replace `CaloDigi/SiWEcalDigi.xml` with our modified file.

```bash
cp SiWEcalDigi.xml ILDConfig/StandardConfig/production/CaloDigi/SiWEcalDigi.xml
```

Let us use `singularity` with one of our `iLCSoft` docker image

```bash
export SINGULARITY_TMPDIR=/your-path/container/tmp/
export SINGULARITY_CACHEDIR=/your-path/container/cache/
cd ILDConfig/StandardConfig/production
singularity shell -H $PWD --bind $(pwd):/home/ilc/data docker://ilcsoft/ilcsoft-centos7-gcc8.2:v02-01-pre bash
source /home/ilc/ilcsoft/v02-01-pre/init_ilcsoft.sh
```

Now run reconstruction algorithm

```bash
Marlin MarlinStdReco.xml \
	--constant.lcgeo_DIR=$lcgeo_DIR \
  --constant.DetectorModel=ILD_l5_o1_v02 \
  --constant.OutputBaseName=wgan_test \
  --constant.RunBeamCalReco=false \
  --MyPfoAnalysis.CollectCalibrationDetails=1 \
  --MyPfoAnalysis.ECalCollectionsSimCaloHit=ECalBarrelSiHits \
  --global.LCIOInputFiles=wgan.slcio

```





