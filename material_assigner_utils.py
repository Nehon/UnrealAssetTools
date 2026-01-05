"""
Material Assigner Utilities for Unreal Engine 5.7

This module provides functions for batch-assigning material instances with
auto-discovered textures to static meshes.

Usage:
    import material_assigner_utils as mau
    mau.run_material_assigner()
"""

import unreal
from typing import Optional


# Default texture patterns for each material parameter
DEFAULT_PATTERNS = {
    "BaseColor": ["_BaseColor", "_Diffuse", "_Albedo", "_Color", "_Base_Color", "_D"],
    "Normal": ["_Normal", "_Nrm", "_N", "_NormalMap"],
    "Roughness": ["_Roughness", "_Rough", "_R"],
    "Metallic": ["_Metallic", "_Metal", "_M"],
    "AO": ["_AO", "_Occlusion", "_AmbientOcclusion", "_O"],
    "Emissive": ["_Emissive", "_Emission", "_E", "_Glow"]
}

# Common parameter names in master materials
MATERIAL_PARAMETER_NAMES = {
    "BaseColor": ["BaseColor", "BaseColorMap", "Base Color", "Diffuse", "Albedo", "Color"],
    "Normal": ["Normal", "NormalMap", "Normal Map"],
    "Roughness": ["Roughness", "RoughnessMap", "Rough"],
    "Metallic": ["Metallic", "Matalness", "MetallicMap", "MetalnessMap", "Metal"],
    "AO": ["AO", "AmbientOcclusion", "OcclusionMap", "Ambient Occlusion", "Occlusion"],
    "Emissive": ["Emissive", "Emission", "EmissiveMap", "EmissionMap", "EmissiveColor"]
}


def get_selected_static_meshes() -> list:
    """Get all currently selected static mesh assets from Content Browser."""
    editor_util = unreal.EditorUtilityLibrary()
    selected_assets = editor_util.get_selected_assets()

    static_meshes = []
    for asset in selected_assets:
        if isinstance(asset, unreal.StaticMesh):
            static_meshes.append(asset)

    return static_meshes


def get_static_meshes_from_folder(folder_path: str, recursive: bool = True) -> list:
    """Get all static mesh assets from a content folder path.

    Args:
        folder_path: Content path like "/Game/Meshes" or "/Game/Assets/Props"
        recursive: Whether to search subdirectories
    """
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    # Normalize path
    if not folder_path.startswith("/"):
        folder_path = "/" + folder_path
    if not folder_path.startswith("/Game"):
        folder_path = "/Game" + folder_path

    # Get all assets in folder
    asset_data_list = asset_registry.get_assets_by_path(
        folder_path,
        recursive=recursive
    )

    static_meshes = []
    for asset_data in asset_data_list:
        if asset_data.asset_class_path.asset_name == "StaticMesh":
            asset = asset_data.get_asset()
            if asset:
                static_meshes.append(asset)

    return static_meshes


def find_textures_in_folder(folder_path: str, recursive: bool = True) -> list:
    """Find all texture assets in a folder.

    Args:
        folder_path: Content path like "/Game/Textures"
        recursive: Whether to search subdirectories
    """
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    # Normalize path
    if not folder_path.startswith("/"):
        folder_path = "/" + folder_path
    if not folder_path.startswith("/Game"):
        folder_path = "/Game" + folder_path

    asset_data_list = asset_registry.get_assets_by_path(
        folder_path,
        recursive=recursive
    )

    textures = []
    for asset_data in asset_data_list:
        class_name = str(asset_data.asset_class_path.asset_name)
        if "Texture" in class_name:
            asset = asset_data.get_asset()
            if asset:
                textures.append(asset)

    return textures


def match_texture_to_type(texture_name: str, patterns: dict = None) -> Optional[str]:
    """Match a texture name to a texture type based on patterns.

    Args:
        texture_name: Name of the texture asset
        patterns: Dict mapping type names to list of patterns. Uses defaults if None.

    Returns:
        The matched type name or None if no match
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    texture_name_lower = texture_name.lower()

    for tex_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if pattern.lower() in texture_name_lower:
                return tex_type

    return None


def find_textures_for_mesh(mesh_name: str, textures: list, patterns: dict = None) -> dict:
    """Find textures that match a mesh name.

    Args:
        mesh_name: Name of the static mesh (e.g., "SM_Chair")
        textures: List of texture assets to search
        patterns: Texture type patterns dict

    Returns:
        Dict mapping texture types to texture assets
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    # Extract base name (remove common prefixes like SM_, T_, etc.)
    base_name = mesh_name
    for prefix in ["SM_", "S_", "Mesh_", "M_"]:
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):]
            break

    base_name_lower = base_name.lower()
    matched_textures = {}

    for texture in textures:
        tex_name = texture.get_name()
        tex_name_lower = tex_name.lower()

        # Check if texture name contains the mesh base name
        # Remove common texture prefixes for comparison
        tex_base = tex_name
        for prefix in ["T_", "Tex_", "TX_"]:
            if tex_base.startswith(prefix):
                tex_base = tex_base[len(prefix):]
                break

        tex_base_lower = tex_base.lower()

        # Check if this texture belongs to this mesh
        if base_name_lower in tex_base_lower or tex_base_lower.startswith(base_name_lower):
            # Determine texture type
            tex_type = match_texture_to_type(tex_name, patterns)
            if tex_type and tex_type not in matched_textures:
                matched_textures[tex_type] = texture

    return matched_textures


def create_material_instance(
    mesh: unreal.StaticMesh,
    master_material: unreal.Material,
    output_folder: Optional[str] = None
) -> Optional[unreal.MaterialInstanceConstant]:
    """Create a material instance for a static mesh.

    Args:
        mesh: The static mesh asset
        master_material: The master/parent material
        output_folder: Optional folder path. If None, uses mesh's folder.

    Returns:
        The created MaterialInstanceConstant or None on failure
    """
    mesh_name = mesh.get_name()
    mesh_path = mesh.get_path_name()

    # Determine output path
    if output_folder:
        if not output_folder.startswith("/"):
            output_folder = "/" + output_folder
        if not output_folder.startswith("/Game"):
            output_folder = "/Game" + output_folder
        package_path = output_folder
    else:
        # Use same folder as mesh
        package_path = "/".join(mesh_path.split("/")[:-1])

    # Generate MI name
    mi_name = f"MI_{mesh_name}"
    if mi_name.startswith("MI_SM_"):
        mi_name = "MI_" + mesh_name[3:]  # Remove SM_ prefix

    full_path = f"{package_path}/{mi_name}"

    # Check if already exists
    existing = unreal.EditorAssetLibrary.does_asset_exist(full_path)
    if existing:
        unreal.log_warning(f"Material instance already exists: {full_path}")
        return unreal.EditorAssetLibrary.load_asset(full_path)

    # Create the material instance
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

    mi_factory = unreal.MaterialInstanceConstantFactoryNew()
    mi_factory.set_editor_property("initial_parent", master_material)

    material_instance = asset_tools.create_asset(
        mi_name,
        package_path,
        unreal.MaterialInstanceConstant,
        mi_factory
    )

    if material_instance:
        unreal.log(f"Created material instance: {full_path}")
    else:
        unreal.log_error(f"Failed to create material instance: {full_path}")

    return material_instance


def find_texture_parameter_name(material_instance: unreal.MaterialInstanceConstant, tex_type: str) -> Optional[str]:
    """Find the actual parameter name in a material for a texture type.

    Args:
        material_instance: The material instance to check
        tex_type: The texture type (e.g., "BaseColor", "Normal")

    Returns:
        The actual parameter name or None if not found
    """
    # Get all texture parameter names from parent
    param_names = []

    # Try to get texture parameter info
    parent = material_instance.get_editor_property("parent")
    if not parent:
        return None

    # Get texture parameter names
    texture_param_infos = unreal.MaterialEditingLibrary.get_texture_parameter_names(parent)

    if tex_type not in MATERIAL_PARAMETER_NAMES:
        return None

    possible_names = MATERIAL_PARAMETER_NAMES[tex_type]

    for param_name in texture_param_infos:
        param_str = str(param_name)
        for possible in possible_names:
            if possible.lower() in param_str.lower():
                return param_str

    return None


def assign_textures_to_material(
    material_instance: unreal.MaterialInstanceConstant,
    textures: dict
) -> int:
    """Assign textures to a material instance.

    Args:
        material_instance: The material instance to modify
        textures: Dict mapping texture types to texture assets

    Returns:
        Number of textures successfully assigned
    """
    assigned_count = 0

    for tex_type, texture in textures.items():
        param_name = find_texture_parameter_name(material_instance, tex_type)

        if param_name:
            success = unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
                material_instance,
                param_name,
                texture
            )
            if success:
                assigned_count += 1
                unreal.log(f"  Assigned {tex_type}: {texture.get_name()} -> {param_name}")
            else:
                unreal.log_warning(f"  Failed to assign {tex_type} to {param_name}")
        else:
            # Try direct parameter name
            direct_names = MATERIAL_PARAMETER_NAMES.get(tex_type, [tex_type])
            for name in direct_names:
                success = unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
                    material_instance,
                    name,
                    texture
                )
                if success:
                    assigned_count += 1
                    unreal.log(f"  Assigned {tex_type}: {texture.get_name()} -> {name}")
                    break

    return assigned_count


def assign_material_to_mesh(mesh: unreal.StaticMesh, material_instance: unreal.MaterialInstanceConstant, slot_index: int = 0) -> bool:
    """Assign a material instance to a static mesh's material slot.

    Args:
        mesh: The static mesh to modify
        material_instance: The material instance to assign
        slot_index: Material slot index (default 0)

    Returns:
        True if successful
    """
    # Get current materials
    static_materials = mesh.get_editor_property("static_materials")

    if slot_index >= len(static_materials):
        unreal.log_error(f"Slot index {slot_index} out of range for {mesh.get_name()}")
        return False

    # Create new static material entry
    new_static_material = unreal.StaticMaterial()
    new_static_material.set_editor_property("material_interface", material_instance)
    new_static_material.set_editor_property("material_slot_name", static_materials[slot_index].material_slot_name)

    # Replace the material at the slot
    static_materials[slot_index] = new_static_material
    mesh.set_editor_property("static_materials", static_materials)

    # Mark package dirty for saving
    unreal.EditorAssetLibrary.save_loaded_asset(mesh)

    unreal.log(f"Assigned {material_instance.get_name()} to {mesh.get_name()} slot {slot_index}")
    return True


def process_static_mesh(
    mesh: unreal.StaticMesh,
    master_material: unreal.Material,
    textures: list,
    patterns: dict = None,
    output_folder: Optional[str] = None,
    assign_to_mesh: bool = True
) -> Optional[unreal.MaterialInstanceConstant]:
    """Process a single static mesh: create MI, find textures, assign everything.

    Args:
        mesh: Static mesh to process
        master_material: Parent material for the MI
        textures: List of available textures
        patterns: Texture matching patterns
        output_folder: Output folder for MI (None = same as mesh)
        assign_to_mesh: Whether to assign MI to the mesh

    Returns:
        The created/modified material instance
    """
    mesh_name = mesh.get_name()
    unreal.log(f"\nProcessing: {mesh_name}")

    # Create material instance
    mi = create_material_instance(mesh, master_material, output_folder)
    if not mi:
        return None

    # Find matching textures
    matched_textures = find_textures_for_mesh(mesh_name, textures, patterns)
    if matched_textures:
        unreal.log(f"  Found {len(matched_textures)} matching textures")
        assign_textures_to_material(mi, matched_textures)
    else:
        unreal.log_warning(f"  No matching textures found for {mesh_name}")

    # Save the material instance
    unreal.EditorAssetLibrary.save_loaded_asset(mi)

    # Assign to mesh if requested
    if assign_to_mesh:
        assign_material_to_mesh(mesh, mi)

    return mi


def run_material_assigner_on_selection(
    master_material_path: str,
    texture_folder: str,
    patterns: dict = None,
    output_folder: Optional[str] = None,
    assign_to_mesh: bool = True
) -> int:
    """Run the material assigner on currently selected static meshes.

    Args:
        master_material_path: Content path to master material
        texture_folder: Content path to texture folder
        patterns: Custom texture patterns (uses defaults if None)
        output_folder: Output folder for MIs (None = same as mesh)
        assign_to_mesh: Whether to assign MI to mesh

    Returns:
        Number of meshes processed
    """
    # Load master material
    master_material = unreal.EditorAssetLibrary.load_asset(master_material_path)
    if not master_material:
        unreal.log_error(f"Could not load master material: {master_material_path}")
        return 0

    # Get selected meshes
    meshes = get_selected_static_meshes()
    if not meshes:
        unreal.log_warning("No static meshes selected in Content Browser")
        return 0

    # Get textures
    textures = find_textures_in_folder(texture_folder)
    unreal.log(f"Found {len(textures)} textures in {texture_folder}")

    # Process each mesh
    processed = 0
    for mesh in meshes:
        if process_static_mesh(mesh, master_material, textures, patterns, output_folder, assign_to_mesh):
            processed += 1

    unreal.log(f"\nCompleted! Processed {processed}/{len(meshes)} meshes")
    return processed


def run_material_assigner_on_folder(
    mesh_folder: str,
    master_material_path: str,
    texture_folder: str,
    patterns: dict = None,
    output_folder: Optional[str] = None,
    assign_to_mesh: bool = True
) -> int:
    """Run the material assigner on all static meshes in a folder.

    Args:
        mesh_folder: Content path to folder containing static meshes
        master_material_path: Content path to master material
        texture_folder: Content path to texture folder
        patterns: Custom texture patterns (uses defaults if None)
        output_folder: Output folder for MIs (None = same as mesh)
        assign_to_mesh: Whether to assign MI to mesh

    Returns:
        Number of meshes processed
    """
    # Load master material
    master_material = unreal.EditorAssetLibrary.load_asset(master_material_path)
    if not master_material:
        unreal.log_error(f"Could not load master material: {master_material_path}")
        return 0

    # Get meshes from folder
    meshes = get_static_meshes_from_folder(mesh_folder)
    if not meshes:
        unreal.log_warning(f"No static meshes found in: {mesh_folder}")
        return 0

    unreal.log(f"Found {len(meshes)} static meshes in {mesh_folder}")

    # Get textures
    textures = find_textures_in_folder(texture_folder)
    unreal.log(f"Found {len(textures)} textures in {texture_folder}")

    # Process each mesh
    processed = 0
    for mesh in meshes:
        if process_static_mesh(mesh, master_material, textures, patterns, output_folder, assign_to_mesh):
            processed += 1

    unreal.log(f"\nCompleted! Processed {processed}/{len(meshes)} meshes")
    return processed


# ============================================================================
# Module Exports
# ============================================================================

# Export convenient names
__all__ = [
    'run_material_assigner_on_selection',
    'run_material_assigner_on_folder',
    'get_selected_static_meshes',
    'get_static_meshes_from_folder',
    'find_textures_in_folder',
    'find_textures_for_mesh',
    'create_material_instance',
    'assign_textures_to_material',
    'assign_material_to_mesh',
    'process_static_mesh',
    'match_texture_to_type',
    'DEFAULT_PATTERNS',
    'MATERIAL_PARAMETER_NAMES'
]
