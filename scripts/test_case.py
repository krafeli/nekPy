from nekPy.launcher.launcher import Launcher
from nekPy.preprocessor.preprocessor import PreProcessor
from nekPy.postprocessor.postprocessor import PostProcessor

preproc = PreProcessor("/home/felix/Projects/thermoSurf/thermoSurf/nekPy/out/test",
                       usr="/home/felix/Projects/thermoSurf/thermoSurf/nekPy/out/loc3.usr",
                       par="/home/felix/Projects/thermoSurf/thermoSurf/nekPy/out/loc3.par",
                       size="/home/felix/Projects/thermoSurf/thermoSurf/nekPy/out/SIZE",
                       additional_files=['/home/felix/run/3d/DU95W180/blasius/xc005/Rek300/probes.xyz'])


"""preproc.generate_bc("/home/felix/run/2d/DU95W180/glob/aoa0/Re1e6/bl/bl.pkl", "blade", 0.05, Lin=15)
preproc.generate_mesh(1., 1., 75, 10, 20, Lin=15)
preproc.msh2nek(periodic_pairs=[(3,4)])

launcher = Launcher("/home/felix/Projects/thermoSurf/thermoSurf/nekPy/out/test")
launcher.makenek()
launcher.run(ncores=4)
"""

postproc = PostProcessor("/home/felix/run/3d/DU95W180/blasius/xc005/Rek300/avgloc30.f00001")

