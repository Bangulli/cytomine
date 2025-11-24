### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib as pl
import xml.etree.ElementTree as ET
### External Imports ###

### Internal Imports ###
from src.utils.switch_case import SwitchCase
########################

REQUIRED_XMLS = [
    ('STAINING', 'staining.xml'),
    ('SAMPLE', 'sample.xml'),
    ('OBSERVATION', 'observation.xml')
]

def get_stain_list(sample, staining, slide_id):
    slide = [f for f in sample.findall('SLIDE') if f.attrib['alias']==slide_id][0]
    stain_list_alias = slide.findall('STAINING_INFORMATION_REF')[0].attrib['alias']
    staining_list = [f for f in staining.findall('STAINING') if f.attrib['alias']==stain_list_alias][0]
    stain_list_meaning = []
    stain_list_code = []
    for stain_elem in staining_list:
        for ca in stain_elem.findall('CODE_ATTRIBUTE'):
            tag_text = ca.findtext('TAG')
            if tag_text == 'staining_compound':
                meaning = ca.findtext('VALUE/MEANING')
                if meaning:
                    stain_list_meaning.append(meaning.lower())
                code = ca.findtext('VALUE/CODE')
                if code:
                    stain_list_code.append(code)
        for sa in stain_elem.findall('STRING_ATTRIBUTE'):
            tag_text = sa.findtext('TAG')
            if tag_text == 'staining_compound':
                val = sa.findtext('VALUE')
                if val:
                    stain_list_meaning.append(val.lower())
    return stain_list_meaning, stain_list_code

def check_stain_ok(sample, staining, slide_id, target) -> bool:
    """Checks if the sample is stained with the "target" compound

    Args:
        sample (ElementTree): ElementTree object, the sample.xml
        staining (ElementTree): ElementTree object, the staining.xml
        slide_id (str): String, the slide id of the sample
        target (str): String, the target value 

    Returns:
        bool: True if stain ok, false if not
    """
    stain_list_meaning, stain_list_code = get_stain_list(sample, staining, slide_id)
    return bool(target in stain_list_meaning + stain_list_code)

def get_specimen_list(sample, slide_id):
    slide = [f for f in sample.findall('SLIDE') if f.attrib['alias']==slide_id][0]
    block_alias = slide.findall('CREATED_FROM_REF')[0].attrib['alias']
    block = [f for f in sample.findall('BLOCK') if f.attrib['alias']==block_alias][0]
    req_specimen = [f.attrib['alias'] for f in block.findall('SAMPLED_FROM_REF')]
    specimen = sample.findall('SPECIMEN')
    block_specimen = [s for s in specimen if s.attrib['alias'] in req_specimen]
    block_biological_being_list= [bb.findall('EXTRACTED_FROM_REF')[0].attrib['alias'] for bb in block_specimen]
    biological_beings = [bb for bb in sample.findall('BIOLOGICAL_BEING') if bb.attrib['alias'] in block_biological_being_list]
    species_meaning = []
    species_code =[]
    for being in biological_beings:
        for ca in being.findall(".//CODE_ATTRIBUTE"):
            tag_text = ca.findtext('TAG')
            if tag_text == 'animal_species':
                meaning = ca.findtext('VALUE/MEANING')
                if meaning:
                    species_meaning.append(meaning.lower())
                code = ca.findtext('VALUE/CODE')
                if code:
                    species_code.append(code)
    return species_code, species_meaning
    
def check_specimen_ok(sample, slide_id, target) -> bool:
    """Checks if the sample is sampled from the "target" species

    Args:
        sample (ElementTree): ElementTree object, the sample.xml
        slide_id (str): String, the slide id of the sample
        target (str): String, the target value 

    Returns:
        bool: True if stain ok, false if not
    """
    species_code, species_meaning = get_specimen_list(sample, slide_id)
    return bool(all([s==target for s in species_meaning]) or all([s==target for s in species_code]))

def get_organ_list(sample, slide_id):
    slide = [f for f in sample.findall('SLIDE') if f.attrib['alias']==slide_id][0]
    block_alias = slide.findall('CREATED_FROM_REF')[0].attrib['alias']
    block = [f for f in sample.findall('BLOCK') if f.attrib['alias']==block_alias][0]
    req_specimen = [f.attrib['alias'] for f in block.findall('SAMPLED_FROM_REF')]
    specimen = sample.findall('SPECIMEN')
    block_specimen = [s for s in specimen if s.attrib['alias'] in req_specimen]
    species_meaning = []
    species_code = []
    for being in block_specimen:
        for code_attr in being.findall(".//CODE_ATTRIBUTE"):
            tag = code_attr.findtext("TAG")
            if tag == "anatomical_site":
                meaning = code_attr.find(".//MEANING").text
                species_meaning.append(meaning.lower())
                code = code_attr.find(".//CODE").text
                species_code.append(code)
    species_meaning=list(set(species_meaning))
    species_code=list(set(species_code))
    return species_meaning, species_code
    
def check_organ_ok(sample, slide_id, target) -> bool:
    """Checks if the sample is sampled from the "target" anatomical site

    Args:
        sample (ElementTree): ElementTree object, the sample.xml
        slide_id (str): String, the slide id of the sample
        target (str): String, the target value 

    Returns:
        bool: True if stain ok, false if not
    """
    organ_meaning, organ_code = get_organ_list(sample, slide_id)
    return bool(target in organ_code or target in organ_meaning) 

def get_case_ID(sample, slide_id):
    slide = [f for f in sample.findall('SLIDE') if f.attrib['alias']==slide_id][0]
    block_alias = slide.findall('CREATED_FROM_REF')[0].attrib['alias']
    block = [f for f in sample.findall('BLOCK') if f.attrib['alias']==block_alias][0]
    req_specimen = [f.attrib['alias'] for f in block.findall('SAMPLED_FROM_REF')]
    specimen = sample.findall('SPECIMEN')
    block_specimen = [s for s in specimen if s.attrib['alias'] in req_specimen]
    block_case_list= [bb.findall('PART_OF_CASE_REF')[0].attrib['alias'] for bb in block_specimen]
    case_list = [bb for bb in sample.findall('CASE') if bb.attrib['alias'] in block_case_list][0].attrib['alias']
    return [case_list]

def get_diagnosis(observation, case_ID):
    obs_list = [o for o in observation.findall('OBSERVATION') if o.findall('CASE_REF')[0].attrib['alias']==case_ID[0]]
    statements = [s.findall('.//CODE_ATTRIBUTE') for s in obs_list[0].findall('STATEMENT') if s.findtext('STATEMENT_TYPE')=='Diagnosis'][0]
    diagnoses = [d.find(".//MEANING").text for d in statements if d.findtext("TAG")=='Diagnosis']
    return diagnoses
    
def get_meta(image_id, slide_id, xmls)->dict:
    stain_list_meaning, stain_list_code = get_stain_list(xmls['SAMPLE'], xmls['STAINING'], slide_id)
    species_code, species_meaning = get_specimen_list(xmls['SAMPLE'], slide_id)
    organ_meaning, organ_code = get_organ_list(xmls['SAMPLE'], slide_id)
    case_ID = get_case_ID(xmls['SAMPLE'], slide_id)
    diagnosis = get_diagnosis(xmls['OBSERVATION'], case_ID)
    return {'staining': stain_list_meaning, 'species': species_meaning, 'organ': organ_meaning, 'case': case_ID, 'diagnosis': diagnosis}

def get_meta_with_codes(image_id, slide_id, xmls)->dict:
    stain_list_meaning, stain_list_code = get_stain_list(xmls['SAMPLE'], xmls['STAINING'], slide_id)
    species_code, species_meaning = get_specimen_list(xmls['SAMPLE'], slide_id)
    organ_meaning, organ_code = get_organ_list(xmls['SAMPLE'], slide_id)
    case_ID = get_case_ID(xmls['SAMPLE'], slide_id)
    diagnosis = get_diagnosis(xmls['OBSERVATION'], case_ID)
    return {'staining': stain_list_meaning, 'species': species_meaning, 'organ': organ_meaning, 'case': case_ID, 'diagnosis': diagnosis, 'staining_code': stain_list_code, 'species_code': species_code, 'organ_code': organ_code, 'case': case_ID, 'diagnosis': diagnosis}

def check_is_include(filter, image_id, slide_id, xmls) -> bool:
    if len(filter)==0:
        return parse_condition(filter, image_id, slide_id, xmls)
    else:
        results = []
        for child in filter:
            tag = child.tag.upper()
            if tag == 'AND':
                results.append(parse_and(child, image_id, slide_id, xmls))
            elif tag == 'OR':
                results.append(parse_or(child, image_id, slide_id, xmls))
            elif tag == 'CONDITION':
                #print("Parsing Condition")
                results.append(parse_condition(child, image_id, slide_id, xmls))
            else:
                raise SyntaxError(f"Unknown child tag name {child.tag}")
        return any(results)
            
def parse_condition(condition, image_id, slide_id, xmls):
    with SwitchCase(condition.attrib['variable'].upper()) as switch:
        if switch.case("STAINING"):
            return bool(check_stain_ok(xmls['SAMPLE'], xmls['STAINING'], slide_id, condition.attrib['value']))
        elif switch.case("SPECIES"):
            return bool(check_specimen_ok(xmls['SAMPLE'], slide_id, condition.attrib['value']))
        elif switch.case("ANATOMICAL_SITE"):
            return bool(check_organ_ok(xmls['SAMPLE'], slide_id, condition.attrib['value']))

def parse_and(filter, image_id, slide_id, xmls):
    #print(f"[{filter.attrib.get('n', '?')}] -> results:", [check_is_include(c, image_id, slide_id, xmls) for c in filter], '->', all([bool(check_is_include(child, image_id, slide_id, xmls)) for child in filter]))
    return all([bool(check_is_include(child, image_id, slide_id, xmls)) for child in filter])

def parse_or(filter, image_id, slide_id, xmls):
    #print(f"[{filter.attrib.get('n', '?')}] -> results:", [check_is_include(c, image_id, slide_id, xmls) for c in filter], '->', any([bool(check_is_include(child, image_id, slide_id, xmls)) for child in filter]))
    return any([bool(check_is_include(child, image_id, slide_id, xmls)) for child in filter])

def load_filter_deps(name):
    image_xml = ET.parse('/inputs/datasets/'+name+'/METADATA/image.xml') if os.path.exists('/inputs') else ET.parse('inputs/datasets/'+name+'/METADATA/image.xml')
    ids = {} ## find all relations between image and slide ids
    for image in image_xml.findall('IMAGE'):
        ids[image.attrib['alias']] = [f.attrib['alias'] for f in list(image.iter()) if f.tag=='IMAGE_OF'][0]
    xmls = {}
    for filter, knowledge_base in REQUIRED_XMLS:
        xmls[filter] = ET.parse(f"/inputs/datasets/{name}/METADATA/{knowledge_base}") if os.path.exists('/inputs') else ET.parse(f"inputs/datasets/{name}/METADATA/{knowledge_base}")
    return ids, xmls