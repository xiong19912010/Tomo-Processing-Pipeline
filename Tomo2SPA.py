#!/usr/bin/python3

from metadata import MetaData, Label, Item
import fire
from fire import core
import logging
import os
import sys
import traceback
import shutil
import imutils
import mrcfile
import starfile
import pandas as pd
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
pd.options.mode.chained_assignment = None

path_current = os.getcwd()
mrc_folder = "Mrc_MotionCor"
mrc_swap_folder = "Mrc_Swap_MotionCor"
stack_folder = "Stack"
ali_folder = "Ali_AreTomo"
ctf_folder = "CTF"
relion_tomo_folder = "Relion_Tomos"
tomogrames_folder = "tomograms"
split_folder = 'split_star'


class Tomo2SPA:
    """
    \n This programe is to systematically analyze tomo data and transfer the information obtained from tomo to SPA \n
    For detailed descreption, please run the following commands (Noted that command stack_spe is optional and only used when log files are missing):
    python3 Tomo2SPA.py motioncor -h
    python3 Tomo2SPA.py stack -h
    python3 Tomo2SPA.py stack_special -h
    python3 Tomo2SPA.py ali -h
    python3 Tomo2SPA.py ctf -h
    python3 Tomo2SPA.py rec -h
    python3 Tomo2SPA.py particle_pick -h
    python3 Tomo2SPA.py coords -h
    python3 Tomo2SPA.py relion_star -h
    python3 Tomo2SPA.py split -h
    python3 Tomo2SPA.py image_average -h
    """

    def motioncor(self, foldername, gain, swap_xy=1, output_star="Mrc_MotionCor.star", image_extension='.tif', frame_number=5, pixel_size=1.35, patch="5 5", kV=300, Ftbin=2, Gpu=1, GpuMemUsage=0.8):
        """
        This command is to do the motionCor for the images\n
        Example:\n
        python3 Tomo2SPA.py motioncor foldername gain [--swap_xy] [--output_star] [--image_extension] [--frame_number] [--pixel_size] [--patch] [--kV] [--Ftbin] [--Gpu] [--GpuMemUsage]
        :param foldername: Directory containing tif images
        :param gain: File containg the machine background information (dm4 file).
        :param swap_xy: 1 means True, 0 means False
        :param output_star: Star file similar to 'relion'.
        :param image_extension: The default extension of image is '.tif'. if not, you can change it via the parameter
        :param frame_number:How many frames in each tif
        :param pixel_size:pixel size in angstroms
        :param path:What step you choose to do the motionCor
        :param Kv:Voltage used to get the images
        :param Ftbin:The compression factor
        :param Gpu:Define the number of Gpu
        """

        MotionCorr2 = "/programs/bin/motioncor2/MotionCor2_1.2.6-Cuda101"
        swap = "mrc_2d_swap_xy_sm7.0_cu9.1"
        alltif = sorted(os.listdir(foldername))
        # dirname = "mrc_motioncor"
        mrc_path = os.path.join(path_current, mrc_folder)
        if os.path.exists(mrc_path):
            shutil.rmtree(mrc_path)
        os.mkdir(mrc_path)
        if gain[-4:] == '.dm4':
            output_gain = gain.replace('.dm4', '.mrc')
            cmd_gain = "dm2mrc "+gain+" "+output_gain
            os.system(cmd_gain)
            gain = output_gain

        mrc_swap_path = os.path.join(path_current, mrc_swap_folder)
        if swap_xy == 1:
            if os.path.exists(mrc_swap_path):
                shutil.rmtree(mrc_swap_path)
            os.mkdir(mrc_swap_path)

        i = 0
        md = MetaData()
        md.addLabels('rlnIndex', 'rlnTifName', 'rlnMrcName', 'rlnPixelSize',
                     'rlnNrOfFrames', 'rlnVoltage')
        for tiff in alltif:
            if tiff[-4:] == image_extension:
                outmrc = tiff.replace(image_extension, '.mrc')
                Input_tif = os.path.join(path_current, foldername, tiff)
                output_mrc = os.path.join(mrc_path, outmrc)
                cmd = MotionCorr2 + " -InTiff " + Input_tif + " -OutMrc " + output_mrc + " -Gain "+gain+" -Patch "+patch +  \
                    " -FtBin "+str(Ftbin)+" -PixSize "+str(pixel_size)+" -kV " + \
                    str(kV)+" -Gpu " + str(Gpu) +  \
                    " -GpuMemUsage "+str(GpuMemUsage)
                os.system(cmd)
                # print(cmd)
                i += 1
                it = Item()
                md.addItem(it)
                md._setItemValue(it, Label('rlnIndex'), str(i))
                md._setItemValue(it, Label('rlnTifName'), Input_tif)
                if swap_xy == 1:
                    swap_output = os.path.join(mrc_swap_path, outmrc)
                    cmd_swap = swap+" "+output_mrc+" "+swap_output
                    os.system(cmd_swap)
                    md._setItemValue(it, Label('rlnMrcName'), swap_output)
                else:
                    md._setItemValue(it, Label('rlnMrcName'), output_mrc)
                md._setItemValue(it, Label('rlnPixelSize'), pixel_size)
                md._setItemValue(it, Label('rlnNrOfFrames'), frame_number)
                md._setItemValue(it, Label('rlnVoltage'), kV)
        md.write(output_star)
        print('Motioncorrection is Done!!!')

    def stack(self, foldername, swap_xy=1, output_star='Stack.star', frame_number=5):
        """
        This command is to stack the mrc files obtained from motionCor\n
        Example:\n
        python3 Tomo2SPA.py stack foldername [--swap_xy] [--output_star] [--frame_number]
        :param foldername:This is the log files
        :param swap_xy: swap the image in xy direction.  1 means True, 0 means false
        :param frame_number:How many frames in each tif


        """

        dos2unix = "dos2unix"
        logfs = sorted(os.listdir(foldername))
        logf_path = os.path.join(path_current, foldername)
        stack_path = os.path.join(path_current, stack_folder)
        if os.path.exists(stack_path):
            shutil.rmtree(stack_path)
        os.mkdir(stack_path)
        tif2st = "tif2st.com"
        tif2st_path = os.path.join(path_current, tif2st)
        with open(tif2st_path, 'w') as f:
            f.write("#!/bin/tcsh -f")
            f.write("\n")
            f.write("\n")
            f.write("set logf=$1")
            f.write("\n")
            f.write("\n")
            f.write("set root=`basename $logf .log`")
            f.write("\n")
            f.write("\n")
            if swap_xy == 1:
                f.write("set alltif = `gawk '/"+str(frame_number)+" frames/||/Saved/' $logf  | gawk '//{if(\""+"Saved"+"\"==$1){ printf(\""+"%s %s\\n" +
                        "\", src[n], $10)}; s=$0; n=split($NF,src,\""+"\\\\"+"\");}' | sort -k 2 -n |gawk '//{printf(\""+mrc_swap_folder+"/%s "+"\", $1)}'`")
            else:
                f.write("set alltif = `gawk '/"+str(frame_number)+" frames/||/Saved/' $logf  | gawk '//{if(\""+"Saved"+"\"==$1){ printf(\""+"%s %s\\n" +
                        "\", src[n], $10)}; s=$0; n=split($NF,src,\""+"\\\\"+"\");}' | sort -k 2 -n |gawk '//{printf(\""+mrc_folder+"/%s "+"\", $1)}'`")
            f.write("\n")
            f.write("\n")
            f.write("set allmrc=`echo $alltif |sed 's/.tif/.mrc/g'`")
            f.write("\n")
            f.write("\n")
            f.write("newstack ${allmrc}  "+stack_folder+"/${root}.mrc")
            f.write("\n")
            f.write("\n")
            f.write("gawk '/"+str(frame_number)+" frames/||/Saved/' $logf  | gawk '//{if(\""+"Saved"+"\"==$1){ printf(\""+"%s %s\\n" +
                    "\", src[n], $10)}; s=$0; n=split($NF,src,\""+"\\\\"+"\");}' | sort -k 2 -n |awk '{print $2}' > "+stack_folder+"/${root}.tlt")
        i = 0
        md = MetaData()
        md.addLabels('rlnIndex', 'rlnTomoTiltSeriesName')
        for logf in logfs:
            if logf[-4:] == ".log":
                Input_logf = os.path.join(logf_path, logf)
                output_st = os.path.join(
                    stack_path, logf).replace('.log', '.mrc')
                # print(Input_logf)
                cmd_dos2unix = dos2unix+" "+Input_logf
                os.system(cmd_dos2unix)
                cmd_chmod = "chmod +x "+tif2st_path
                os.system(cmd_chmod)
                cmd = tif2st_path+" "+Input_logf
                # print(cmd)
                os.system(cmd)
                i += 1
                it = Item()
                md.addItem(it)
                md._setItemValue(it, Label('rlnIndex'), str(i))
                md._setItemValue(it, Label('rlnTomoTiltSeriesName'), output_st)
        md.write(output_star)
        print('MRC to Stack files is Done!!!')

    def stack_special(self, foldername, tomo_column_number=3, angle_column_number=5):
        """
        \nThis command is to stack the mrc files without log files\n
        Example:\n
        python3 Tomo2SPA.py stack_spe foldername [--swap_xy] [--tomo_column_number] [--angle_column_number]
        :param foldername:It contains the mrc files after motioncor.
        :param tomo_column_number:The image name must have tomo name information. After splitting the name by "_" separator, the column number of the tomo.
        :param angle_column_number: The image name must have angle information. After splitting the name by "_" separator, the column number of the angle.

        """
        newstack = "newstack"
        stack_path = os.path.join(path_current, stack_folder)
        if os.path.exists(stack_path):
            shutil.rmtree(stack_path)
        os.mkdir(stack_path)
        mrc_path = os.path.join(path_current, foldername)
        stack_names = sorted(os.listdir(foldername))
        stack_number = []
        for stack_name in stack_names:
            stack_number.append(stack_name.split('_')[tomo_column_number][:])
        # print(stack_number)
        from iteration_utilities import unique_everseen
        from iteration_utilities import duplicates
        dupes = list(unique_everseen(duplicates(stack_number)))
        for i in range(0, len(dupes)):
            filter_stack_no_path = []
            for stack_name in stack_names:
                if stack_name.split('_')[tomo_column_number][:] == dupes[i]:
                    # input_mrc = os.path.join(mrc_path, stack_name)
                    filter_stack_no_path.append(stack_name)
            filter_stack_no_path.sort(key=lambda x: float(
                x.split('_')[angle_column_number][:]))
            filter_stack = []
            for k in range(0, len(filter_stack_no_path)):
                input_mrc = os.path.join(mrc_path, filter_stack_no_path[k])
                filter_stack.append(input_mrc)
            filter_stack = ' '.join(filter_stack)
            output_name = "".join(stack_name.split(
                '_')[0]+dupes[i]+'.mrc')
            output_path = os.path.join(stack_path, output_name)
            cmd = newstack+" "+filter_stack+" "+output_path

           # print(cmd)
            os.system(cmd)
            print()

    def ali(self, foldername=stack_folder, output_star='Ali_AreTomo.star', alignZ=200, volz=0, outbin=8, pixsize=2.7, TiltCor=0, outxf=1, gpu=1):
        """
        This command is to align the stack using AreTomo\n
        Example:\n
        python3 Tomo2SPA ali [--foldername] [--output_star] [--alignZ] [--volz] [--outbin] [--pixsize] [--outxf] [--gpu]

        :param foldername:The stack files

        :param alignZ: Volume height for alignment

        :param volz:Volume z height for reconstrunction. It must be  greater than 0 to reconstruct a volume.Default is 0, only aligned tilt series will generated.

        :param outBin: Binning for aligned output tilt series
        :param Gpu:GPU IDs.

        :param pixsize:1. Pixel size in Angstrom of the input tilt series. It is only required for dose weighting. If missing, dose
        weighting will be disabled.
        :param TiltCor:Whether to apply the offset of tilt angle.
        :param outxf:1. When set by giving no-zero value, IMOD compatible XF file will be generated.



        """
        aretomo = "AreTomo_1.0.6_Cuda102"

        stacks = sorted(os.listdir(foldername))
        stack_path = os.path.join(path_current, foldername)
        alignment_path = os.path.join(path_current, ali_folder)
        if os.path.exists(alignment_path):
            shutil.rmtree(alignment_path)
        os.mkdir(alignment_path)

        i = 0
        md = MetaData()
        md.addLabels('rlnIndex', 'rlntlt', 'rlnxf', 'rlnali')

        for stack in stacks:
            if stack[-4:] == '.mrc':
                input_stack = os.path.join(stack_path, stack)
                input_tlt = os.path.join(
                    stack_path, stack).replace('.mrc', '.tlt')
                tlt_data = open(input_tlt, 'r').read().split('\n')
                data = pd.to_numeric(pd.DataFrame(tlt_data)[0])
                max_tlt = round(data.max())
                min_tlt = round(data.min())
                tiltrange = str(min_tlt)+' '+str(max_tlt)
                output_ali = os.path.join(
                    alignment_path, stack).replace('.mrc', '.ali')
                cmd = aretomo+" -InMrc "+input_stack+" -OutMrc "+output_ali + " -TiltRange " + tiltrange+" -AlignZ " + \
                    str(alignZ)+" -VolZ " + str(volz)+" -OutBin " + str(outbin) + \
                    " -PixSize " + str(pixsize)+" -OutXF " + \
                    str(outxf) + " -TiltCor " + \
                    str(TiltCor) + " -Gpu "+str(gpu)
                os.system(cmd)
                output_tlt = os.path.join(
                    alignment_path, stack).replace('.mrc', '.ali.tlt')
                output_xf = os.path.join(
                    alignment_path, stack).replace('.mrc', '.xf')
                i += 1
                it = Item()
                md.addItem(it)
                md._setItemValue(it, Label('rlnIndex'), str(i))
                md._setItemValue(it, Label('rlntlt'), output_tlt)
                md._setItemValue(it, Label('rlnxf'), output_xf)
                md._setItemValue(it, Label('rlnali'), output_ali)
        md.write(output_star)
        print('Alignment using Aretomo is Done!!!')

    def ctf(self, foldername=stack_folder, output_star='ctf.star', pixsize=2.7, kV=300, spheriacal_abrreation=2.7, amplitude_contrast=0.1, spectrum_size=1024, Min_resolution=30, Max_resolution=10, Min_defocus=30000, Max_defocus=70000, defocus_search_step=100):
        """
        This command is to do CTF calculation using CTFFind4\n
        Example:\n
        python3 Tomo2SPA.py ctf [--foldername] [--output_star] [--pixsize] [--kV] [--spheriacal_abrreation] [--amplitude_contrast] [--spectrum_size] [--Min_resolution] [--Max_defocus] [--Min_defocus] [--Max_defocus] [--defocus_search_step]
        :param foldername: The stack files
        :param output_star
        :param pixsize:Pixel size
        :param kV:Acceleration voltage
        :param spheriacal_abrreation:
        :param amplitude_contrast
        :param spectrum_size:Size of amplitude spectrum to compute
        :param Min_resolution:Minimum resolution of spectrum to caclulate
        :param Max_resolution:Maximum resolution of spectrum to calculate
        :param Min_defocus:Minimum defocus in angstroms, e.g 30000
        :param Max_defocus:Maximum defocus in angstroms, e.g 70000
        :param defocus_search_step:Defocus search step
        """

        ctf_pkg = "ctffind4.sh"
        ctf_pkg_path = os.path.join(path_current, ctf_pkg)
        with open(ctf_pkg_path, 'w') as f:
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("\n")
            f.write("mrcfile=$*")
            f.write("\n")
            f.write("path_current=$(pwd)")
            f.write("\n")
            f.write("fname=$(basename $mrcfile)")
            f.write("\n")
            f.write("echo Processing $mrcfile")
            f.write("\n")
            f.write("comfile=$path_current/"+ctf_folder +
                    "/${fname%.mrc}_ctffind4.com")
            f.write("\n")
            f.write("echo \"" + "#!/bin/bash"+"\" > $comfile")
            f.write("\n")
            f.write(
                "echo \"" + "ctffind > $path_current/"+ctf_folder+"/${fname%.mrc}.log << EOF"+"\" >>  $comfile")
            f.write("\n")
            f.write("echo  $mrcfile >>  $comfile")
            f.write("\n")
            f.write("echo no >>  $comfile")
            f.write("\n")
            f.write(
                "echo $path_current/"+ctf_folder+"/${fname%.mrc}_output.ctf >> $comfile")
            f.write("\n")
            f.write("echo "+str(pixsize)+" >> $comfile")
            f.write("\n")
            f.write("echo "+str(kV)+" >> $comfile")
            f.write("\n")
            f.write("echo "+str(spheriacal_abrreation)+" >> $comfile")
            f.write("\n")
            f.write("echo "+str(amplitude_contrast)+" >> $comfile")
            f.write("\n")
            f.write("echo "+str(spectrum_size)+" >> $comfile")
            f.write("\n")
            f.write("echo "+str(Min_resolution)+" >> $comfile")
            f.write("\n")
            f.write("echo "+str(Max_resolution)+" >> $comfile")
            f.write("\n")
            f.write("echo "+str(Min_defocus)+" >> $comfile")
            f.write("\n")
            f.write("echo "+str(Max_defocus)+" >> $comfile")
            f.write("\n")
            f.write("echo "+str(defocus_search_step)+" >> $comfile")
            f.write("\n")
            f.write("echo no >> $comfile")
            f.write("\n")
            f.write("echo no >> $comfile")
            f.write("\n")
            f.write("echo no >> $comfile")
            f.write("\n")
            f.write("echo no >> $comfile")
            f.write("\n")
            f.write("echo no >> $comfile")
            f.write("\n")
            f.write("echo \"" + "EOF"+"\" >> $comfile")
            f.write("\n")
            f.write("echo exit 0 >> $comfile")
            f.write("\n")
            f.write("bash  $comfile")
            f.write("\n")
        cmd_chmod = "chmod +x "+ctf_pkg_path
        os.system(cmd_chmod)

        stacks = sorted(os.listdir(foldername))
        stack_path = os.path.join(path_current, foldername)
        ctf_path = os.path.join(path_current, ctf_folder)
        if os.path.exists(ctf_path):
            shutil.rmtree(ctf_path)
        os.mkdir(ctf_path)

        i = 0
        md = MetaData()
        md.addLabels('rlnIndex', 'rlnTomoImportCtfFindFile')
        for stack in stacks:
            if stack[-4:] == '.mrc':
                input_stack = os.path.join(
                    stack_path, stack)
                output_ctf = os.path.join(
                    ctf_path, stack).replace('.mrc', '_output.txt')
                cmd = ctf_pkg_path+" "+input_stack
                os.system(cmd)
                print(cmd)
                i += 1
                it = Item()
                md.addItem(it)
                md._setItemValue(it, Label('rlnIndex'), str(i))
                md._setItemValue(
                    it, Label('rlnTomoImportCtfFindFile'), output_ctf)
        md.write(output_star)
        print('CTF calculation is Done!!!')

    def rec(self, foldername=stack_folder, output_star='rec.star', total_images=35, Imagesize='3838 3710', imagesAreBinned=1, binByFactor=8, radial="0.35,0.035", gpu=0, thickness=200, rotateby=90):
        """"
        \nThis command is to build the reconstruction using IMOD\n
        Example:\n
        python3 Tomo2SPA.py rec [--foldername] [--output_star] [--total_images] [--Imagesize] [--imagesAreBinned] [--binByFactor] [--radial] [--gpu] [--thickness] [--rotateby] 
        :param foldername: Stack files
        :param total_images: How many images in each stack
        :param binByFactor
        :param thickness:Thickness of the reconstruction after binning



        """
        newstack = "newstack"
        tilt = "tilt"
        stacks = sorted(os.listdir(foldername))
        stack_path = os.path.join(path_current, foldername)
        # dirname = "relion_tomos"
        relion_tomos_path = os.path.join(path_current, relion_tomo_folder)
        if os.path.exists(relion_tomos_path):
            shutil.rmtree(relion_tomos_path)
        os.mkdir(relion_tomos_path)
        # dirname1 = "tomogrames"
        tomo_path = os.path.join(relion_tomos_path, tomogrames_folder)
        if os.path.exists(tomo_path):
            shutil.rmtree(tomo_path)
        os.mkdir(tomo_path)
        i = 0
        md = MetaData()
        md.addLabels('rlnIndex', 'rlnrec', 'rlnBinedmrc')
        for stack in stacks:
            if stack[-4:] == '.mrc':
                input_name = os.path.basename(stack)
                output_folder_name = input_name[:-4]
                output_folder_path = os.path.join(
                    tomo_path, output_folder_name)
                if os.path.exists(output_folder_path):
                    shutil.rmtree(output_folder_path)
                os.mkdir(output_folder_path)
                input_stack = os.path.join(
                    stack_path, stack)
                outputFile = input_name.replace(
                    '.mrc', '_bin' + str(binByFactor) + '_ali.mrc')
                outputFile_path = os.path.join(output_folder_path, outputFile)
                rec_name = input_name.replace(
                    '.mrc', '_bin'+str(binByFactor)+'.rec')
                rec_output = os.path.join(output_folder_path, rec_name)
                rec_sirt_name = input_name.replace(
                    '.mrc', '_bin'+str(binByFactor)+'_sirt_tmp.rec')
                rec_sirt_output = os.path.join(
                    output_folder_path, rec_sirt_name)
                rec_sirt_final_name = input_name.replace(
                    '.mrc', '_bin'+str(binByFactor)+'_sirt.rec')
                rec_sirt_final_output = os.path.join(
                    output_folder_path, rec_sirt_final_name)

                transformFile = input_name.replace('.mrc', '.xf')
                tiltFile = input_name.replace('.mrc', '.ali.tlt')
                tiltFile_modified = tiltFile.replace('.ali.tlt', '.tlt')
                ctffile = input_name.replace('.mrc', '_output.txt')
                transformFile_path = os.path.join(
                    path_current, ali_folder, transformFile)
                cmd_cp_xf = "cp "+transformFile_path+" "+output_folder_path
                os.system(cmd_cp_xf)
                # print(cmd_cp_xf)
                tiltFile_path = os.path.join(
                    path_current, ali_folder, tiltFile)
                cmd_cp_tilt = "cp "+tiltFile_path+" " + \
                    os.path.join(output_folder_path, tiltFile_modified)
                os.system(cmd_cp_tilt)
                # print(cmd_cp_tilt)
                ctffile_path = os.path.join(path_current, ctf_folder, ctffile)
                cmd_cp_ctf = "cp "+ctffile_path+" "+output_folder_path
                os.system(cmd_cp_ctf)
                # print(cmd_cp_ctf)
                cmd_link_mrc = "ln -s "+input_stack+" "+output_folder_path
                os.system(cmd_link_mrc)

                cmd_newstack = newstack+" "+"-InputFile"+" "+input_stack+" "+"-OutputFile"+" "+outputFile_path+" " + \
                    "-TransformFile"+" "+transformFile_path+" " + \
                    "-ImagesAreBinned"+" " + \
                    str(imagesAreBinned)+" " + \
                    "-BinByFactor"+" "+str(binByFactor)
                # print(cmd_newstack)
                os.system(cmd_newstack)
                cmd_tilt = tilt+" "+"-input"+" "+outputFile_path+" "+"-output"+" "+rec_output+" -TILTFILE"+" "+tiltFile_path + \
                    " "+"-RADIAL"+" "+radial+" "+"-UseGPU " + \
                        str(gpu)+" "+"-THICKNESS "+str(thickness) + \
                        " -RotateBy "+str(rotateby)
                print(cmd_tilt)
                os.system(cmd_tilt)

                # sirt reconstruction
                cmd_tilt_sirt = tilt+" "+"-input"+" "+outputFile_path+" "+"-output"+" "+rec_sirt_output+" -TILTFILE"+" "+tiltFile_path + \
                    " "+"-RADIAL"+" "+radial+" "+"-UseGPU " + \
                        str(gpu)+" "+"-THICKNESS "+str(thickness) + \
                        " -SIRTIterations 5" + " -StartingIteration 1" + " -MASK 2" + " -LOG 0.0"
                os.system(cmd_tilt_sirt)
                cmd_rotx = "clip" + " rotx " + rec_sirt_output + " " + rec_sirt_final_output
                os.system(cmd_rotx)
                print(cmd_tilt_sirt)
                xtilt_name = input_name.replace('.mrc', '.xtilt')
                xtilt_path = os.path.join(output_folder_path, xtilt_name)
                with open(xtilt_path, 'w') as f:
                    k = 0
                    for k in range(0, total_images):
                        m = 0.00
                        f.write(str(m))
                        f.write("\n")

                newst_name = "newst.com"
                newst_path = os.path.join(output_folder_path, newst_name)
                with open(newst_path, 'w') as f:
                    f.write("$setenv IMOD_OUTPUT_FORMAT MRC")
                    f.write("\n")
                    f.write("$newstack -StandardInput")
                    f.write("\n")
                    f.write("AntialiasFilter    -1")
                    f.write("\n")
                    f.write("InputFile    "+input_name)
                    f.write("\n")
                    f.write("OutputFile    "+outputFile)
                    f.write("\n")
                    f.write("TransformFile    " + transformFile)
                    f.write("\n")
                    f.write("TaperAtFill    1,1")
                    f.write("\n")
                    f.write("AdjustOrigin")
                    f.write("\n")
                    f.write("OffsetsInXandY    0.0,0.0")
                    f.write("\n")
                    f.write("ImagesAreBinned    "+str(imagesAreBinned))
                    f.write("\n")
                    f.write("BinByFactor    "+str(binByFactor))
                    f.write("\n")
                    f.write("$if (-e ./savework) ./savework")

                tilt_name = "tilt.com"
                tilt_path = os.path.join(output_folder_path, tilt_name)
                with open(tilt_path, 'w') as f:
                    f.write("$setenv IMOD_OUTPUT_FORMAT MRC")
                    f.write("\n")
                    f.write("$tilt -StandardInput")
                    f.write("\n")
                    f.write("InputProjections " + outputFile)
                    f.write("\n")
                    f.write("OutputFile "+rec_name)
                    f.write("\n")
                    f.write("IMAGEBINNED " + str(binByFactor))
                    f.write("\n")
                    f.write("TILTFILE "+tiltFile_modified)
                    f.write("\n")
                    f.write("THICKNESS " + str(thickness*binByFactor))
                    f.write("\n")
                    f.write("RADIAL "+radial)
                    f.write("\n")
                    f.write("XAXISTILT 0.0")
                    f.write("\n")
                    f.write("RotateBy"+str(rotateby))
                    f.write("\n")
                    f.write("MODE 2")
                    f.write("\n")
                    f.write("FULLIMAGE "+Imagesize)
                    f.write("\n")
                    f.write("SUBSETSTART 0,0")
                    f.write("\n")
                    f.write("AdjustOrigin")
                    f.write("\n")
                    f.write("ActionIfGPUFails 1,2")
                    f.write("\n")
                    f.write("XTILTFILE " + xtilt_name)
                    f.write("\n")
                    f.write("OFFSET 0.0")
                    f.write("\n")
                    f.write("SHIFT 0.0 0.0")
                    f.write("\n")
                    f.write("UseGPU "+str(gpu))
                    f.write("\n")
                    f.write("$if (-e ./savework) ./savework")

                i += 1
                it = Item()
                md.addItem(it)
                md._setItemValue(it, Label('rlnIndex'), str(i))
                md._setItemValue(it, Label('rlnrec'), rec_output)
                md._setItemValue(it, Label('rlnBinedmrc'), outputFile_path)
        md.write(output_star)
        print('tomo reconstruction is Done!!!')

    def particle_pick(self, foldername=relion_tomo_folder, reference=str(), rec_file=str(), total_partile=5000, distance_threshold=25, cc=1.5, cc_up=6):
        """"
         \nThis command is to do particle picking using eman2\n
         Example:\n
         python3 Tomo2SPA.py particle_pick [--foldername]  [--reference] [--rec_file] [--total_partile] [--distance_threshold] [--cc]
         :param foldername: relions folder
         :param reference: this is the templace map you used
         :param rec_file: this is extension name you will deal with
         :param total_partile: set a threshold for how many paritles you want to produce in each sample
         :param distance_threshold:distance threshold for each particle
         :param cc: this is the cross correlation threshold value. the bigger it is, the less particle you will get but with more accuracy.



         """
        cmd_source = "source  /programs/setup/EMAN2_setup.csh"
        eman = 'e2spt_tempmatch_edit.py'
        os.system(cmd_source)
        tomo_path = os.path.join(
            path_current, foldername, tomogrames_folder)
        tomos = sorted(os.listdir(tomo_path))
        for tomo in tomos:
            path_rec = os.path.join(tomo_path, tomo)
            recs = os.listdir(path_rec)
            for rec in recs:
                if rec[-len(rec_file):] == rec_file:
                    print('processing...........'+rec)
                    input_rec = os.path.join(path_rec, rec)
                    outputname = rec.replace('.rec', '.rec.txt')
                    output_path = os.path.join(path_rec, outputname)
                    output_mod = rec.replace('.rec', '_pick.mod')
                    output_mod_path = os.path.join(
                        path_rec, output_mod)
                    reference_path = os.path.join(
                        path_current, reference)
                    cmd = eman+" "+input_rec+" "+"--reference "+reference_path+" --nptcl " + \
                        str(total_partile)+" --dthr " + \
                        str(distance_threshold) + \
                        " --vthr "+str(cc) + " --vthr_up " + \
                        str(cc_up) + " --rmedge"
                    # print(cmd)
                    os.system(cmd)
                    cmd_awk = "awk '{print $3*2,$2*2,$1*2}' " + \
                        output_path+" > tmp1.txt"
                    # print(cmd_awk)
                    os.system(cmd_awk)
                    cmd_mod = "point2model tmp1.txt "+output_mod_path
                    # print(cmd_mod)
                    os.system(cmd_mod)
                    print(rec+' particle picking is done !!!!')
        print('All particle picking is Done!!!')

    def coords(self, foldername=relion_tomo_folder, output_star='Tomo_Coords.star', coord_file=str(), binsize=8):
        """"
        \nThis command is to prepare the particle coordinates for relion\n
        Example:\n
        python3 Tomo2SPA.py coords [--foldername] [--output_star] [--binsize] [--coord_file]

        :param foldername: reconstrunction folder where it contains particle coordinates.
        :param coord_file: this is extension name you will deal with

        """
        model2point = "model2point"
        # dirname = "tomogrames"
        foldername1 = os.path.join(path_current, foldername, tomogrames_folder)
        tomos = sorted(os.listdir(foldername1))
        tomos_folder_path = foldername1
        i = 0
        md = MetaData()
        md.addLabels('rlnTomoName', 'rlnTomoImportParticleFile')

        for tomo in tomos:
            # input_mod_name = os.path.join(tomo, '.mod')
            input_mod_folder = os.path.join(tomos_folder_path, tomo)
            mods = sorted(os.listdir(input_mod_folder))
            for mod in mods:
                if mod[-len(coord_file):] == coord_file:

                    input_mod_path = os.path.join(tomos_folder_path, tomo, mod)
                    output_txt_name = mod.replace('.mod', '.txt')
                    output_txt_path = os.path.join(
                        tomos_folder_path, tomo, output_txt_name)
                    cmd = model2point+" "+input_mod_path+" "+"-contour "+output_txt_path
                    os.system(cmd)
                    coor_star = output_txt_path.replace('.txt', '.star')
                    coords = open(output_txt_path)
                    lines = coords.read().splitlines()
                    dataframe_coord = pd.DataFrame(
                        lines, columns=['coordinates'])
                    new = dataframe_coord['coordinates'].str.split(
                        " +", n=4, expand=True)
                    dataframe_coord['rlnHelicalTubeID'] = pd.to_numeric(
                        new[1], errors='coerce')
                    dataframe_coord['rlnCoordinateX'] = pd.to_numeric(
                        new[2], errors='coerce')*binsize
                    dataframe_coord['rlnCoordinateY'] = pd.to_numeric(
                        new[3], errors='coerce')*binsize
                    dataframe_coord['rlnCoordinateZ'] = pd.to_numeric(
                        new[4], errors='coerce')*binsize
                    dataframe_coord = dataframe_coord.loc[(
                        dataframe_coord != 0).all(axis=1)]
                    dataframe_coord.drop(
                        columns=['coordinates'], inplace=True)

                    starfile.write(dataframe_coord, coor_star, overwrite=True)
                    i += 1
                    it = Item()
                    md.addItem(it)
                   # md._setItemValue(it, Label('rlnIndex'), str(i))
                    md._setItemValue(it, Label('rlnTomoName'), tomo)
                    md._setItemValue(
                        it, Label('rlnTomoImportParticleFile'), coor_star)
        output_star = os.path.join(path_current, foldername, output_star)
        md.write(output_star)
        print('model to points is Done!!!')

    def relion_star(self, foldername=relion_tomo_folder, output_star='Tomos_Descr.star', step=3):
        """
        \nThis command is to prepare the tomo description star file\n

        Example:\n
        python3 Tomo2SPA.py relion_star [--foldername] [--output_star] [--step]
        :param foldername: Relion_tomos
        :param step: the step angle

        Default value


        """

        relion_path = os.path.join(path_current, foldername)
        csv_folder = 'csv_all'
        csv_folder_path = os.path.join(relion_path, csv_folder)
        if os.path.exists(csv_folder_path):
            shutil.rmtree(csv_folder_path)
        os.mkdir(csv_folder_path)
        import csv
        import numpy as np
        import pandas as pd
        i = -1
        md = MetaData()
        md.addLabels('rlnTomoName', 'rlnTomoTiltSeriesName', 'rlnTomoImportCtfFindFile',
                     'rlnTomoImportImodDir', 'rlnTomoImportOrderList')
        tomos_path = os.path.join(relion_path, tomogrames_folder)
        tomos = sorted(os.listdir(tomos_path))
        for tomo in tomos:
            print('processing........'+tomo)
            all_files = os.listdir(os.path.join(tomos_path, tomo))
            for file in all_files:
                if file[-4:] == '.tlt':
                    file_path = os.path.join(
                        tomos_path, tomo, file)
                    tlt = open(file_path, 'r')
                    data1 = tlt.read().split('\n ')
                    b = pd.DataFrame(data1)[0]
                    data = pd.to_numeric(b)
                    output_name = file.replace('.tlt', '.csv')
                    csv_output_path = os.path.join(
                        csv_folder_path, output_name)
                    with open(csv_output_path, 'w') as f:
                        writer = csv.writer(f, delimiter=',')
                        max_angle = int(data.max())
                        min_angle = int(data.min())
                        index = len(data)
                        positive_angle = range(0, max_angle+step, step)
                        negative_angle = range(-step, min_angle-step, -step)
                        angle = list(positive_angle)+list(negative_angle)
                        i = 0
                        for j in range(0, index):
                            writer.writerow([j, angle[j]])

            i += 1
            tomo_mrc_name = "".join(tomo+'.mrc')
            tomo_mrc_path = os.path.join(tomos_path, tomo, tomo_mrc_name)
            tomo_ctf_name = "".join(tomo+'_output.txt')
            tomo_ctf_path = os.path.join(tomos_path, tomo, tomo_ctf_name)
            it = Item()
            md.addItem(it)
            md._setItemValue(it, Label('rlnIndex'), str(i))
            md._setItemValue(it, Label('rlnTomoName'), tomo)

            md._setItemValue(it, Label('rlnTomoTiltSeriesName'),
                             tomo_mrc_path)
            md._setItemValue(
                it, Label('rlnTomoImportCtfFindFile'), tomo_ctf_path)
            md._setItemValue(it, Label('rlnTomoImportImodDir'),
                             os.path.join(tomos_path, tomo))
            md._setItemValue(
                it, Label('rlnTomoImportOrderList'), csv_output_path)
        output_star = os.path.join(relion_path, output_star)
        md.write(output_star)
        print('relion_star preparation is Done!!!')

    def split(self, files, pixsize=2.7, binsize=8):
        '''
        pyhton3 Tomo2SPA.py split file_name [--pixsize] [--binsize]
        :param file_name: this is the particle.star file
        :param pixsize:this is the orignal pixsize of mrc
        :param binsize: this is the binfactor in reconstruction files
        '''
        print('Processing ................')

        star_spa_all = pd.DataFrame([starfile.read(files)]).iloc[0, 1]
        Tomonames = star_spa_all['rlnTomoName'].unique()
        after_spa_path = os.path.join(path_current, split_folder)
        if os.path.exists(after_spa_path):
            shutil.rmtree(after_spa_path)
        os.mkdir(after_spa_path)
        for Tomoname in Tomonames:
            output_name = Tomoname+'_split.star'
            output_txt = Tomoname+'.txt'
            output_mod = Tomoname+'.mod'
            output_path = os.path.join(after_spa_path, output_name)
            output_txt_path = os.path.join(after_spa_path, output_txt)
            output_mod_path = os.path.join(after_spa_path, output_mod)
            star_spa = star_spa_all[star_spa_all['rlnTomoName'] == Tomoname]
            star_spa['rlnCoordinateX'] = (
                star_spa['rlnCoordinateX']-star_spa['rlnOriginXAngst']/pixsize)/binsize
            star_spa['rlnCoordinateY'] = (
                star_spa['rlnCoordinateY']-star_spa['rlnOriginYAngst']/pixsize)/binsize
            star_spa['rlnCoordinateZ'] = (
                star_spa['rlnCoordinateZ']-star_spa['rlnOriginZAngst']/pixsize)/binsize
            star_spa['rlnOriginXAngst'] = 0
            star_spa['rlnOriginYAngst'] = 0
            star_spa['rlnOriginZAngst'] = 0
            np.savetxt(output_txt_path, star_spa[['rlnHelicalTubeID',
                                                  'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']], fmt='%d %1.3f %1.3f %1.3f')
            cmd = 'point2model '+output_txt_path+' '+output_mod_path
            os.system(cmd)
            starfile.write(star_spa, output_path, overwrite=True)
        print('Splitting work is done!!!!!!!!!')

    def image_average(self, particle_star_file, image_folder):
        '''
        python3 Tomo2SPA.py image_average [particle_star_file] [image_folder]
        :param particle_star_file: This is the [particles.star] generated by Relion
        :param image_folder: This is the [Subtomograms]
        '''
        # calculate the rotation matrix between two 3D vectors. Pay attention here that we only care about the angles so we should normalize the vectors first.
        def Rotation_matrix(p0, p1):
            from numpy import linalg as LA
            # two random 3D vectors
            p0 = np.array(p0)
            p1 = np.array(p1)
            # calculate cross and dot products
            C = np.cross(p0, p1)
            D = np.dot(p0, p1)
            NP0 = LA.norm(p0)  # used for scaling
            j = 0
            for i in range(0, len(C)):
                if C[i] == 0:
                    j = j+1
            if j < len(C):  # check for colinearity
                Z = np.array(
                    [[0, -C[2], C[1]], [C[2], 0, -C[0]], [-C[1], C[0], 0]])
                R = (np.eye(3) + Z + np.dot(Z, Z) * (1-D) /
                     (LA.norm(C)**2)) / NP0**2  # rotation matrix
            else:
                R = np.sign(D) * (LA.norm(p1) / NP0)  # orientation and scaling
                R = np.eye(3)
            return R

        # Rotate Image based on Rotation Matrix
        def Image_rotation_3D(array, orient):
            rot = orient[1]
            tilt = orient[2]
            phi = orient[0]

            # create meshgrid
            dim = array.shape
            ax = np.arange(dim[0])
            ay = np.arange(dim[1])
            az = np.arange(dim[2])
            coords = np.meshgrid(ax, ay, az)

            # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
            xyz = np.vstack([coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
                             # y coordinate, centered
                             coords[1].reshape(-1) - float(dim[1]) / 2,
                             coords[2].reshape(-1) - float(dim[2]) / 2])  # z coordinate, centered

            # create transformation matrix
            r = R.from_euler('xyz', [rot, tilt, phi], degrees=True)
            mat = r.as_matrix()

            # apply transformation
            transformed_xyz = np.dot(mat, xyz)

            # extract coordinates
            x = transformed_xyz[0, :] + float(dim[0]) / 2
            y = transformed_xyz[1, :] + float(dim[1]) / 2
            z = transformed_xyz[2, :] + float(dim[2]) / 2

            x = x.reshape((dim[1], dim[0], dim[2]))
            y = y.reshape((dim[1], dim[0], dim[2]))
            # I test the rotation in 2D and this strange thing can be explained
            z = z.reshape((dim[1], dim[0], dim[2]))
            new_xyz = [y, x, z]
            arrayR = map_coordinates(array, new_xyz, order=1)
            return arrayR

        # create a folder to store the averaged mrcs
        print('Processing ................')
        folder_name = 'Subtomograms_Average'
        star_file_name = 'new_particles.star'
        output_average_mrc_folder = os.path.join(path_current, folder_name)
        if os.path.exists(output_average_mrc_folder):
            shutil.rmtree(output_average_mrc_folder)
        os.mkdir(output_average_mrc_folder)
        ## import particles_star_file
        particle_star = starfile.read(particle_star_file)
        particle_star = pd.DataFrame([particle_star]).iloc[0, 1]
        subtomogram_names = particle_star['rlnTomoName'].unique()
        #subtomogram_names = ['SpermTS_002', 'SpermTS_003']

        # create a new star file to store the information
        New_particle_star = pd.DataFrame()
        j = 0
        for subtomogram_name in subtomogram_names:
            print('processing '+subtomogram_name+'.........')
            split_star = pd.DataFrame()
            split_star = particle_star[particle_star['rlnTomoName']
                                       == subtomogram_name].reset_index(drop=True)
            # sorted the coordinates and split out the image names
            Imagenames = split_star['rlnImageName']
            Imagenames_split = Imagenames.str.split('/', expand=True)
            Imagenames_only = Imagenames_split.iloc[:,
                                                    Imagenames_split.shape[1]-1]
            split_star['Imagename'] = Imagenames_only
            test = Imagenames_only.str.split('_', expand=True)
            test[0] = pd.to_numeric(test[0])
            split_star['Drop'] = test[0]
            split_star = split_star.sort_values(
                by=['Drop']).reset_index(drop=True)
            split_star = split_star.drop(columns=['Drop'])

            # deal with each subtomograms
            IDs = split_star['rlnHelicalTubeID'].unique()

            for ID in IDs:
                j += 1
                print('processing '+subtomogram_name+'  TubeID '+str(ID))
                # split the star file again based on TubeID

                xyz = pd.DataFrame()
                xyz = split_star[split_star['rlnHelicalTubeID']
                                 == ID].reset_index(drop=True)

                # set the first 2 mrcs as the reference
                mrc_file_0 = xyz['Imagename'][0]
                mrc_file_0_path = os.path.join(
                    image_folder, subtomogram_name, mrc_file_0)
                total_mrc_0 = np.array(mrcfile.open(mrc_file_0_path).data)
                mrc_file_1 = xyz['Imagename'][1]
                mrc_file_1_path = os.path.join(
                    image_folder, subtomogram_name, mrc_file_1)
                total_mrc_1 = np.array(mrcfile.open(mrc_file_1_path).data)
                #total_mrc = total_mrc_0+total_mrc_1
                total_mrc = np.zeros(
                    [total_mrc_0.shape[0], total_mrc_0.shape[1], total_mrc_0.shape[2]], dtype=np.int16)

                for i in range(1, xyz.shape[0]):
                    mrc_file = xyz['Imagename'][i]
                    mrc_file_path = os.path.join(
                        image_folder, subtomogram_name, mrc_file)
                    mrc_p1 = np.array(mrcfile.open(mrc_file_path).data)

                    p_ref = xyz[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].iloc[1,
                                                                                             :]-xyz[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].iloc[0, :]

                    p_ref[2] = 0
                    # p_ref=np.array([0,1,0])
                    p_ref = p_ref / np.sqrt(np.sum(p_ref**2))
                    p1 = xyz[['rlnCoordinateX', 'rlnCoordinateY',
                              'rlnCoordinateZ']].iloc[i-1, :]
                    p2 = xyz[['rlnCoordinateX', 'rlnCoordinateY',
                              'rlnCoordinateZ']].iloc[i, :]
                    p_21 = p2-p1
                    p_21 = p_21 / np.sqrt(np.sum(p_21**2))
                    # calculate rotation matrix and euler angle
                    m = Rotation_matrix(p_ref, p_21)
                    mat = R.from_matrix(m)
                    orient = mat.as_euler('xyz', degrees=True)

                    # rotate image
                    mrc_p1_rotate = Image_rotation_3D(mrc_p1, orient)
                    # averge image
                    total_mrc = mrc_p1_rotate+total_mrc

                output_name = subtomogram_name+'_'+str(ID)+'_ID_average.mrc'
                output_path = os.path.join(
                    output_average_mrc_folder, output_name)
                with mrcfile.new(output_path, overwrite=True) as mrc:
                    mrc.set_data(total_mrc/xyz.shape[0])
                intermid_star = pd.DataFrame(xyz.iloc[0, :]).T
                intermid_star = intermid_star.drop(
                    columns=['Imagename'])
                intermid_star = intermid_star.drop(
                    columns=['rlnHelicalTubeID'])
                if (j % 2) == 0:
                    intermid_star['rlnRandomSubset'] = 2
                else:
                    intermid_star['rlnRandomSubset'] = 1
                intermid_star['rlnImageName'] = output_path
                New_particle_star = New_particle_star.append(
                    intermid_star, ignore_index=True)
        output_star_path = os.path.join(path_current, star_file_name)
        starfile.write(New_particle_star, output_star_path, overwrite=True)
        print('Averaging work is done!!!!!!!!!')


def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)


if __name__ == "__main__":
    core.Display = Display
    fire.Fire(Tomo2SPA)
