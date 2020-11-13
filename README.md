# Generation of showers

We would like to generate showers with our generative models (preferably with GPUs) and create `lcio` files. In this way, we could use `iLCsoft` ecosystem to use these showers for reconstruction and further anaylsis.

## Local
```
docker run -it -v ~/synthetic-data-generator:/home/ilc/datagen --network host ilcsoft/py3lcio:lcio-15-04_v2 bash
pip install streamlit
streamlit run generator.py -- --output wgantest.slcio --nbsize 100 --nevents 50
```

