import fitz
import tabula
import os
from pdf2image import convert_from_path
import json
from openai import OpenAI
import re
import sys
from dotenv import load_dotenv

# File paths
folder_path = "mpea_data/part1"
failed_files_log = "database_method/o3mini_multiple_request_no_source_text/again/failed_files.txt"
log_file_path = "database_method/o3mini_multiple_request_no_source_text/again/process_log.txt"

# Logger for stdout redirection
class Logger:
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, "w")
        self.prompt_file = open(filepath.replace(".txt", "_prompts.txt"), "w")

    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        
        if any(keyword in message for keyword in ["Prompt:", "Response:", "Input Text:"]):
            self.prompt_file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()
        self.prompt_file.flush()
        
    def close(self):
        self.file.close()
        self.prompt_file.close()

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class ExtractAgent:
    def __init__(self, client):
        self.client = client
        
    def ensure_folder_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
    def extract_images(self, pdf_path, output_folder):
        self.ensure_folder_exists(output_folder)
        images = []
        pdf = fitz.open(pdf_path)

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                image_path = os.path.join(output_folder, image_name)

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                images.append(image_path)
        return images

    def extract_tables(self, pdf_path):
        tables = []
        try:
            dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            for i, df in enumerate(dfs):
                if not df.empty:
                    tables.append({
                        "table_index": i,
                        "data": df.to_dict(orient="records")
                    })
        except Exception as e:
            print(f"Error extracting tables: {e}")
        return tables

    def extract_text(self, pdf_path):
        pdf = fitz.open(pdf_path)
        text_data = []

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()
            text_data.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })
        return text_data
        
    def extract_from_pdf(self, file_path):
            output_folder = os.path.join(os.path.dirname(file_path), "pdf_extract")
            self.ensure_folder_exists(output_folder)
            pdf_name = os.path.basename(file_path).replace('.pdf', '')
            
            result = {
                "file_name": pdf_name,
                "images": self.extract_images(file_path, os.path.join(output_folder, f"{pdf_name}_images")),
                "tables": self.extract_tables(file_path),
                "text": self.extract_text(file_path)
            }
            
            output_json_path = os.path.join(output_folder, f"{pdf_name}_result.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
                
            all_text = []
            for page in result["text"]:
                all_text.append(f"Page {page['page_number']}:\n{page['text']}")
                
            # add content for tables
            for table in result["tables"]:
                all_text.append(f"\nTable {table['table_index']}:\n{json.dumps(table['data'], indent=2)}")
                

            all_text.append("\nImages found:")
            for img_path in result["images"]:
                all_text.append(f"- {os.path.basename(img_path)}")
                
            combined_text = "\n".join(all_text)
            
            # print("\nExtraction Result:")
            # print(combined_text)
            return combined_text


class JsonFixAgent:
    def __init__(self, client):
        self.client = client
    
    def fix_json(self, raw_text, schema):
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
            
        system_prompt = f"""
        You are a JSON fixing assistant. Convert the provided raw text into a structured JSON object.
        Each material should be an individual entry in the database with its own composition and processing data.
        IMPORTANT:
        1. Create separate entries for each material variation
        2. Preserve ALL information from the input. You can use 'Null' for missing values.
        3. Ensure composition values are numerical when possible (remove 'at%' and convert to numbers)
        4. Never discard any information
        
        Schema: {json.dumps(schema, indent=4)}
        """
        
        print("\nJSON Fixing Prompt:")
        print(system_prompt)
        print("\nInput Text:")
        print(raw_text)
        
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_text}
                ],
                # temperature=0
                reasoning_effort="high"
            )
            
            content = response.choices[0].message.content
            print("\nJSON Fixing Response:")
            print(content)
            
            json_block = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            json_str = json_block.group(1).strip() if json_block else content.strip()
            json_str = re.sub(r'\s*//.*$', '', json_str, flags=re.MULTILINE)
            
            parsed = json.loads(json_str)
            return [parsed] if isinstance(parsed, dict) else parsed
            
        except Exception as e:
            print(f"JSON fixing error: {str(e)}")
            return None


class ValidationAgent:
    def __init__(self, client):
        self.client = client
    
    def validate(self, file_path, article_database):
        extract_agent = ExtractAgent(self.client)
        
        # 先提取 PDF 全文
        full_text = extract_agent.extract_from_pdf(file_path)
        
        if not full_text:
            print("Validation extraction failed, returning original database")
            return article_database

        # 2. For each material, ensure all phases' volume percentages sum to ~100%.
        validation_prompt = f"""
        You are a materials database validator. Review and correct this database:
        {json.dumps(article_database, indent=2)}

        Check and correct:
        1. Extract each material's data at each distinct processing state as a separate entry.
        2. Only extract the information of the complete material.
        3. Match properties with their corresponding processing conditions.
        4. For each material, please check phases' volume percentages are right.
        5. Do not combine data from different processing states.
        
        Use the original PDF content to verify the correctness of the structured data.
        
        Return the corrected database in the EXACT SAME FORMAT as the input.
        Maintain the exact same JSON structure.
        Do not add any commentary or explanation - just return the corrected JSON.

        Below is the extracted text from the PDF file:
        {full_text}
        """


        try:
            print("\nCalling OpenAI API for validation...")

            response = self.client.chat.completions.create(
                model="o3-mini", 
                messages=[
                    {"role": "system", "content": validation_prompt},
                    {"role": "user", "content": full_text}
                ],
                # temperature=0
                reasoning_effort="high",
            )

            content = response.choices[0].message.content
            
            print("\nValidation Prompt:")
            print(validation_prompt.replace(full_text, "[FULL_TEXT_TRUNCATED]")) 
            print("\nValidation Response:")
            print(content)


            json_block = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            json_str = json_block.group(1).strip() if json_block else content.strip()

            try:
                validated_data = json.loads(json_str)
                return validated_data
            except json.JSONDecodeError:
                print("Failed to parse validator output, returning original database")
                return article_database

        except Exception as e:
            print(f"Validation error: {str(e)}")
            return article_database

# Schemas definition
composition_processing_schema = {
    "Material": "any",
    "Composition": {
        "Fe": "number or string or null",
        "Ni": "number or string or null",
        "Co": "number or string or null",
        "Mn": "number or string or null",
        "Cr": "number or string or null",
        "Al": "number or string or null",
        "Ti": "number or string or null",
        "Cu": "number or string or null",
        "Si": "number or string or null",
        "V": "number or string or null",
        "Nb": "number or string or null",
        "B": "number or string or null",
        "Mo": "number or string or null",
        "Ta": "number or string or null",
        "Other": "object or null"
    },
    "Processing": {
        "Zero_Processing": "any",
        "Homogenization": "any",
        "Homogenization_Temperature": "number or string or null",
        "Homogenization_Time": "number or string or null",
        "Rolling": "any",
        "Rolling_Temperature": "number or string or null",
        "Rolling_Percent": "number or string or null",
        "Recrystallization": "any",
        "Recrystallization_Temperature": "number or string or null",
        "Recrystallization_Time": "number or string or null",
        "Aging": "any",
        "Aging_Temperature": "number or string or null",
        "Aging_Time": "number or string or null",
        "Additional_Processing": "object or null"
    },
}

phases_schema = {
    "Matrix": {
        "Type": "any",
        "Matrix_Volume_Percentage": "number or string or null",
    },
    "First_Precipitate": {
        "Type": "any",
        "Precipitate_Size": "number or string or null",
        "Precipitate_Volume_Percentage": "number or string or null",
    },
    "Second_Precipitate": {
        "Type": "any",
        "Precipitate_Size": "number or string or null",
        "Precipitate_Volume_Percentage": "number or string or null",
    },
    "Third_Precipitate": {
        "Type": "any",
        "Precipitate_Size": "number or string or null",
        "Precipitate_Volume_Percentage": "number or string or null",
    },
    "Additional_Phases": "array or null"
}

properties_schema = {
    "Room_Temperature": {
        "Ultimate_Tensile_Strength": "number or string or null",
        "Ultimate_Compressive_Strength": "number or string or null",
        "Tensile_Yield_Strength": "number or string or null",
        "Compressive_Yield_Strength": "number or string or null",
        "Hardness": "number or string or null",
        "Tensile_Ductility": "number or string or null",
        "Compressive_Ductility": "number or string or null",
        "Additional_Properties": "object or null"
    },
    "Non_Room_Temperature": {
        "Cryo_HT_Strength_Temperature": "number or string or null",
        "Cryo_HT_Strength": "number or string or null",
        "Cryo_HT_Ductility": "number or string or null",
        "Temperature_Dependent_Properties": "array or null",
        "Additional_Properties": "object or null"
    },
    "Other_Properties": "object or null"
}

# Prompt generator functions
def get_composition_prompt(full_text):
    return f"""
    Extract only the **overall composition** (at%) and processing of the complete material from the extracted full text, **excluding** any local phase precipitate, phases, matrix or regional compositions (such as dendrites, inter-dendrites, precipitates, or specific phases like BCC/FCC). 

    ### Instructions:
    When extracting overall composition:
    - Focus on the authors' contributions, avoiding context about prior works.
    - Extract the composition of the complete material rather than localized phases or regions.
    - Treat any difference in **Composition**, **Processing**, **Phases**, or **Properties** as defining a new material.
    - Ignore any composition values given specifically for **phases, dendrites, inter-dendrites, precipitates, or other substructures**.
    - The composition should be reported in atomic percent (at%) and should include the following elements, if available: Fe, Ni, Co, Mn, Cr, Al, Ti, Cu, Si, V, Nb, B, Mo, Ta.
    - If there is no experimentally measured composition for the complete material, provide the **thermodynamic (CALPHAD-predicted) composition** instead.

    When extracting processing, please include the following details (Keep units when extracting):  
    - Homogenization? (Yes=1, No=0)  
    - Homogenization Temperature (°C)  
    - Homogenization time (hr)  
    - Rolling? (Yes=1, No=0)  
    - Rolling Temperature (°C): If it is room temperature rolling, set as 20.  
    - Rolling Percentage (%)
    - Recrystallization? (Yes=1, No=0)  
    - Recrystallization Temperature (°C)  
    - Recrystallization Time (min)  
    - Aging? (Yes=1, No=0)  
    - Aging Temperature (°C)  
    - Aging time (hr)  

    ### Full Text:
    {full_text}

    """

def get_phases_prompt(material_name, existing_info, full_text):
    return f"""
    You are an expert in materials science. Based on the given PDF text, extract the "Phases" section describing the microstructural phases for each processing condition.

    ### Material Information:
    - **Material Name**: {material_name}
    - **Existing Data**:
    {existing_info}

    ### Instructions:
    Based on these information, extract the "Phases" section for '{material_name}', describing the microstructural phases for each processing condition.
    Include details as below (Keep units when extracting):
    - **Matrix**:
    - Type (matrix crystal structure or lattice type, classify: face-centered cubic = 1, body-centered cubic = 2, L12 = 3, B2 = 4, σ = 5)
    - Matrix Volume Percentage (%)
    - **The First Type of Precipitate**:
    - Type (precipitate crystal structure or lattice type, classify as: L12 = 1, γ'' = 2, B2 = 3, η = 4, L21 = 5, σ = 6, BCC = 7, FCC = 8, μ = 9, HCP = 10, Laves = 11)
    - Precipitate Size (nm)
    - Precipitate Volume Percentage (%)
    - **The Second Type of Precipitate**:
    - Type (None = 0, L12 = 1, γ'' = 2, B2 = 3, η = 4, L21 = 5, σ = 6, BCC = 7, FCC = 8, μ = 9, HCP = 10, Laves = 11)
    - Precipitate Size (nm)
    - Precipitate Volume Percentage (%)
    - **The Third Type of Precipitate**:
    - Type (None = 0, L12 = 1, γ'' = 2, B2 = 3, η = 4, L21 = 5, σ = 6, BCC = 7, FCC = 8, μ = 9, HCP = 10, Laves = 11)
    - Precipitate Size (nm)
    - Precipitate Volume Percentage (%)

    ### PDF Full Text:
    {full_text}

    """

# - **Hardness (Vickers Hardness, HV)**: For hardness values derived from tensile strength, explicitly note this. 改成了**Hardness (Vickers Hardness, HV)**
def get_properties_prompt(material_name, existing_info, full_text):
    return f"""
    You are an expert in materials science. Based on the given full text, extract all relevant material properties.
    Based on the existing information for material '{material_name}':

    ### Material Information:
    - **Material Name**: {material_name}
    - **Existing Data**:
    {existing_info}


    ### Instructions:
    Provide the following material properties for material '{material_name}' (Keep units when extracting):
    #### **Measured at Room Temperature**
    - **Ultimate Tensile Strength (MPa)**
    - **Ultimate Compressive Strength (MPa)**
    - **Tensile Yield Strength (MPa)**
    - **Compressive Yield Strength (MPa)**
    - **Hardness (Vickers Hardness, HV)**
    - **Tensile Ductility (%)**
    - **Compressive Ductility (%)**

    #### **Measured at Non-Room Temperatures**
    - **Cryogenic or High-Temperature Strength Temerature (°C)**.
    - **Cryogenic or High-Temperature Strength (MPa)**
    - **Cryogenic or High-Temperature Ductility (%)**

    ### Full Text:
    {full_text}

    """

def find_sets(obj, path="root"):
    """Recursively search for `set` in a JSON structure"""
    if isinstance(obj, set):
        print(f"Found set at {path}: {obj}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            find_sets(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            find_sets(item, f"{path}[{i}]")

def convert_sets_to_lists(obj):
    """Recursively convert sets in an object to lists."""
    if isinstance(obj, dict):
        # If it is a dictionary, recursively convert the value
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # If it is a list, recursively convert each element
        return [convert_sets_to_lists(v) for v in obj]
    elif isinstance(obj, set):
        # If it is a set, convert it to a list
        return list(obj)
    else:
        return obj


def process_file(file_path, client):
    extract_agent = ExtractAgent(client)
    json_fix_agent = JsonFixAgent(client)
    validation_agent = ValidationAgent(client)

    file_name = os.path.basename(file_path).replace(".pdf", "")
    article_database = {}

    # Step 1: extract full text of pdf
    print(f"Extracting full text from {file_name}...")
    full_text = extract_agent.extract_from_pdf(file_path)

    if not full_text:
        print("Failed to extract content, skipping file.")
        return None

    print("\nExtraction Result:")
    print(full_text)

    # Step 2: extract Composition & Processing Prompt
    print("Extracting composition and processing...")
    composition_prompt = get_composition_prompt(full_text)

    print("\nComposition & Processing Prompt:")
    print(composition_prompt.replace(full_text, "[FULL_TEXT_TRUNCATED]"))

    response = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "system", "content": composition_prompt},
            {"role": "user", "content": "Please extract the composition and processing details from full text."}
        ],
        # temperature=0
        reasoning_effort="high",
    )
    composition_raw_text = response.choices[0].message.content

    print("\nComposition & Processing Response:")
    print(composition_raw_text)

    structured_data = json_fix_agent.fix_json(composition_raw_text, composition_processing_schema)

    if structured_data:
        for entry in structured_data:
            material_id = entry.get("Material", f"{file_name}_Sample_{len(article_database) + 1}")
            article_database[material_id] = {"composition_processing": entry}

    # Step 3: extract Phases
    print("Extracting phases...")
    for material_id, material_data in article_database.items():
        phases_prompt = get_phases_prompt(
            material_id,
            json.dumps(material_data, indent=4),
            # material_data.get("composition_processing", {}).get("Composition_Source_Text", ""),
            # material_data.get("composition_processing", {}).get("Processing_Source_Text", ""),
            full_text
        )


        print(f"\nPhases Prompt for {material_id}:")
        print(phases_prompt.replace(full_text, "[FULL_TEXT_TRUNCATED]"))


        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": phases_prompt},
                {"role": "user", "content": "Please extract the phases details from full text."}
            ],
            # temperature=0
            reasoning_effort="high",
        )
        phases_raw_text = response.choices[0].message.content
        phases_data = json_fix_agent.fix_json(phases_raw_text, phases_schema)

        if phases_data:
            material_data["phases"] = phases_data[0] if isinstance(phases_data, list) else phases_data

        # Step 4: extract Properties
        print("Extracting properties...")
        properties_prompt = get_properties_prompt(
            material_id,
            json.dumps(material_data, indent=4),
            # material_data.get("composition_processing", {}).get("Composition_Source_Text", ""),
            # material_data.get("composition_processing", {}).get("Processing_Source_Text", ""),
            # material_data.get("phases", {}).get("Phases_Source_Text", ""),
            full_text
        )

        print(f"\nProperties Prompt for {material_id}:")
        print(properties_prompt.replace(full_text, "[FULL_TEXT_TRUNCATED]")) 

        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": properties_prompt},
                {"role": "user", "content": "Please extract the properties details from full text."}
            ],
            # temperature=0
            reasoning_effort="high",
        )
        properties_raw_text = response.choices[0].message.content
        properties_data = json_fix_agent.fix_json(properties_raw_text, properties_schema)

        if properties_data:
            material_data["properties"] = properties_data[0] if isinstance(properties_data, list) else properties_data

    # Step 5: save data
    extracted_file = f"{file_name}_extracted.json"
    extracted_path = os.path.join("database_method/o3mini_multiple_request_no_source_text/again/", extracted_file)
    with open(extracted_path, "w") as f:
        json.dump(article_database, f, indent=4)

    print(f"Saved extracted data to {extracted_file}")


    # Step 6: confirm data
    print(f"Validating {extracted_file}...")
    validated_database = validation_agent.validate(file_path, article_database)

    validated_file = f"{file_name}_validated.json"
    validated_path = os.path.join("database_method/o3mini_multiple_request_no_source_text/again/", validated_file)
    with open(validated_path, "w") as f:
        json.dump(validated_database, f, indent=4)

    print(f"Saved validated data to {validated_file}")

def main():

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Initialize logging
    logger = Logger(log_file_path)
    sys.stdout = logger
    failed_files = []
    
    # Process all PDF files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"\nProcessing {filename}")
            try:
                process_file(file_path, client)
            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")
                failed_files.append(filename)

    # Log failed files
    with open(failed_files_log, "w") as f:
        f.write("\n".join(failed_files))

    print("\nProcessing completed.")
    logger.close()
    sys.stdout = logger.console

if __name__ == "__main__":
    main()