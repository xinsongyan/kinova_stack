import os
import xml.etree.ElementTree as ET

urdf_folder = "kinova_description/urdf"
print(f"Current working directory: {urdf_folder}")

for filename in os.listdir(urdf_folder):
    if filename.endswith(".urdf"):
        urdf_path = os.path.join(urdf_folder, filename)
        print(f"Processing {urdf_path}")

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()


            # Ensure the root tag is <robot>
            if root.tag != "robot":
                print(f"  Skipped {urdf_path}: root tag is not <robot>")
                continue
            
            # Replace all ".dae" with ".STL" in mesh filenames
            for mesh in root.iter('mesh'):
                fname = mesh.get('filename')
                if fname and fname.endswith('.dae'):
                    new_fname = fname[:-4] + '.STL'
                    mesh.set('filename', new_fname)
                    print(f"  Changed {fname} to {new_fname}")

            # Define the mujoco element
            mujoco_elem = ET.Element("mujoco")
            compiler_elem = ET.SubElement(mujoco_elem, "compiler")
            compiler_elem.set("meshdir", "../meshes/")
            compiler_elem.set("balanceinertia", "true")
            compiler_elem.set("discardvisual", "false")

            # Add a blank line after the mujoco block (as a comment node)
            mujoco_tail = ET.Comment(" ")
            mujoco_elem.tail = "\n\n"
            
            # Check if mujoco tag already exists
            if root.find("mujoco") is None:
                root.insert(0, mujoco_elem)
                print("  Inserted mujoco block.")

            tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
            print(f"  Updated {urdf_path}")
        except Exception as e:
            print(f"  Failed to process {urdf_path}: {e}")