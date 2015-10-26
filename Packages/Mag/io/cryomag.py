__author__ = 'wack'


import xml.etree.ElementTree as ET
from RockPy3.core import io

class CryoMag(io.ftype):
    def __init__(self, dfile, dialect=None):
        tree = ET.parse(dfile)

        specimens = {}

        for specimen in tree.getroot().findall("specimen"):

            # read in specimen properties
            specimens[specimen.attrib['name']] = {}
            specimens[specimen.attrib['name']]['coreaz'] = float(specimen.attrib['coreaz'])
            specimens[specimen.attrib['name']]['coredip'] = float(specimen.attrib['coredip'])
            specimens[specimen.attrib['name']]['beddip'] = float(specimen.attrib['beddip'])
            specimens[specimen.attrib['name']]['bedaz'] = float(specimen.attrib['bedaz'])
            specimens[specimen.attrib['name']]['vol'] = float(specimen.attrib['vol'])
            specimens[specimen.attrib['name']]['weight'] = float(specimen.attrib['weight'])

            for step in specimen.findall("step"):
                # append all steps from the file
                stepdata = {}

                # get data for one step
                stepdata['step'] = step.attrib['step']
                stepdata['type'] = step.attrib['type']
                stepdata['comment'] = step.attrib['comment']

                # read measurements of this step
                stepdata['measurements'] = []
                for magmoment in step.findall("measurements/magmoment"):
                    measurement_data = {}
                    measurement_data['name'] = magmoment.attrib['type']
                    measurement_data['values'] = {}
                    measurement_data['values']['X'] = float(magmoment.attrib['X'])
                    measurement_data['values']['Y'] = float(magmoment.attrib['Y'])
                    measurement_data['values']['Z'] = float(magmoment.attrib['Z'])
                    measurement_data['values']['I'] = float(magmoment.attrib['I'])
                    measurement_data['values']['D'] = float(magmoment.attrib['D'])
                    measurement_data['values']['M'] = float(magmoment.attrib['M'])
                    measurement_data['values']['sM'] = float(magmoment.attrib['sM'])
                    measurement_data['values']['a95'] = float(magmoment.attrib['a95'])
                    measurement_data['values']['time'] = magmoment.attrib['time']

                    # in files before version 1.4 attribute no_readings does not exist
                    # use default of three which was always used
                    try:
                        no_readings = int(magmoment.attrib['no_readings'])
                    except KeyError:
                        measurement_data['values']['no_readings'] = 3
                    else:
                        measurement_data['values']['no_readings'] = no_readings

                    stepdata['measurements'].append(measurement_data)

                results = step.find('results')

                stepdata['results'] = {}
                stepdata['results']['X'] = float(results.attrib['X'])
                stepdata['results']['Y'] = float(results.attrib['Y'])
                stepdata['results']['Z'] = float(results.attrib['Z'])
                stepdata['results']['I'] = float(results.attrib['I'])
                stepdata['results']['D'] = float(results.attrib['D'])
                stepdata['results']['M'] = float(results.attrib['M'])
                stepdata['results']['sM'] = float(results.attrib['sM'])
                stepdata['results']['a95'] = float(results.attrib['a95'])
                stepdata['results']['time'] = results.attrib['time']

                # if we do not have a holder measurement
                # look for holder data that was substracted from the measurement values
                if specimen.attrib['name'] != 'holder':
                    holderresults = step.find('holder')

                    stepdata['holderresults'] = {}
                    stepdata['holderresults']['X'] = float(holderresults.attrib['X'])
                    stepdata['holderresults']['Y'] = float(holderresults.attrib['Y'])
                    stepdata['holderresults']['Z'] = float(holderresults.attrib['Z'])
                    stepdata['holderresults']['I'] = float(holderresults.attrib['I'])
                    stepdata['holderresults']['D'] = float(holderresults.attrib['D'])
                    stepdata['holderresults']['M'] = float(holderresults.attrib['M'])
                    stepdata['holderresults']['sM'] = float(holderresults.attrib['sM'])
                    stepdata['holderresults']['a95'] = float(holderresults.attrib['a95'])
                    stepdata['holderresults']['time'] = holderresults.attrib['time']

                # add stepdata for current step
                specimens['stepdata'] = stepdata

        self.raw_data = specimens

# for initial testing
# delete when no longer needed
if __name__ == '__main__':
    c = CryoMag( dfile= '/Users/mike/Dropbox/experimental_data/RelPint/Step1B/IG_1291A.cmag.xml')
    print(c.raw_data)