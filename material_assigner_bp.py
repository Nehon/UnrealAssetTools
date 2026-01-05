"""
Material Assigner - Blueprint Integration

This module is designed to be called from Unreal Blueprints using
the "Execute Python Command" node.

Supports automatic ORM/ARM packed texture detection:
- ORM/ARM textures: Occlusion(R), Roughness(G), Metallic(B) - SUPPORTED
- RMA textures: Roughness(R), Metallic(G), AO(B) - SKIPPED (incompatible)

Usage from Blueprint (Execute Python Command node):

    # AUTO-DISCOVERY MODE (Recommended - uses mesh material dependencies):
    import material_assigner_bp as mabp; mabp.psa("/Game/Materials/M_Master")
    import material_assigner_bp as mabp; mabp.psa("/Game/Materials/M_Master", "/Game/Materials/M_Master_ORM")
    import material_assigner_bp as mabp; mabp.psa("/Game/Materials/M_Master", "/Game/Materials/M_Master_ORM", overwrite=True)

    # FOLDER SEARCH MODE (searches texture folder by name):
    import material_assigner_bp as mabp; mabp.ps("/Game/Materials/M_Master", "/Game/Textures")
    import material_assigner_bp as mabp; mabp.ps("/Game/Materials/M_Master", "/Game/Textures", "/Game/Materials/M_Master_ORM")

    # PROCESS FOLDER:
    import material_assigner_bp as mabp; mabp.pfa("/Game/Meshes", "/Game/Materials/M_Master")
    import material_assigner_bp as mabp; mabp.pfa("/Game/Meshes", "/Game/Materials/M_Master", "/Game/Materials/M_Master_ORM")
    import material_assigner_bp as mabp; mabp.pf("/Game/Meshes", "/Game/Materials/M_Master", "/Game/Textures")
    import material_assigner_bp as mabp; mabp.pf("/Game/Meshes", "/Game/Materials/M_Master", "/Game/Textures", "/Game/Materials/M_Master_ORM")

Options:
    master_material_orm_path: ORM master material (optional, uses "ORM Map" parameter)
    output_folder: Where to save MIs (default: next to mesh)
    assign_to_mesh: Assign MI to mesh's material slot (default: True)
    overwrite: Delete and recreate existing MIs (default: False)

Shorthand aliases: ps, pf, psa, pfa

The results are printed to the Output Log.
"""

import unreal
from typing import Optional


# =============================================================================
# Default texture patterns
# =============================================================================

DEFAULT_PATTERNS = {
    "BaseColor": ["_BaseColor", "_Diffuse", "_Albedo", "_Color", "_Base_Color", "_D", "_diff"],
    "Normal": ["_Normal", "_Nrm", "_N", "_NormalMap", "_norm"],
    "Roughness": ["_Roughness", "_Rough", "_R", "_roughness"],
    "Metallic": ["_Metallic", "_Metal", "_M", "_metallic"],
    "AO": ["_AO", "_Occlusion", "_AmbientOcclusion", "_O", "_ao"],
    "Emissive": ["_Emissive", "_Emission", "_E", "_Glow", "_emissive"]
}

# Patterns for ORM/ARM packed textures (COMPATIBLE - same channel order: Occlusion/AO(R), Roughness(G), Metallic(B))
ORM_PATTERNS = ["_ORM", "_ARM", "_OcclusionRoughnessMetallic", "_AORoughnessMetallic"]

# Patterns for RMA packed textures (INCOMPATIBLE - different channel order: Roughness(R), Metallic(G), AO(B))
RMA_PATTERNS = ["_RMA", "_RoughnessMetallicAO", "_RoughnessMetalAO"]

MATERIAL_PARAMETER_NAMES = {
    "BaseColor": ["BaseColor", "Base Color", "Diffuse", "Albedo", "Color"],
    "Normal": ["Normal", "NormalMap", "Normal Map"],
    "Roughness": ["Roughness", "Rough"],
    "Metallic": ["Metallic", "Metal"],
    "AO": ["AO", "AmbientOcclusion", "Ambient Occlusion", "Occlusion"],
    "Emissive": ["Emissive", "Emission", "EmissiveColor"]
}

# Parameter names for ORM master material
MATERIAL_PARAMETER_NAMES_ORM = {
    "BaseColor": ["BaseColor", "Base Color", "Diffuse", "Albedo", "Color"],
    "Normal": ["Normal", "NormalMap", "Normal Map"],
    "ORM": ["ORM Map", "ORM", "ARM", "OcclusionRoughnessMetallic", "PackedTexture"],
    "Emissive": ["Emissive", "Emission", "EmissiveColor"]
}


# =============================================================================
# Material Compatibility Checking
# =============================================================================

def get_base_material(material) -> Optional[unreal.Material]:
    """Recursively get the base Material from a Material or MaterialInstance.

    Args:
        material: A Material or MaterialInstance asset

    Returns:
        The base Material, or None if not found
    """
    if material is None:
        return None

    # If it's already a Material (not an instance), return it
    if isinstance(material, unreal.Material):
        return material

    # If it's a MaterialInstance, get parent and recurse
    if isinstance(material, unreal.MaterialInstance):
        parent = material.get_editor_property("parent")
        return get_base_material(parent)

    return None


def get_material_shading_info(material: unreal.Material) -> dict:
    """Get shading-related properties from a Material.

    Args:
        material: A Material asset

    Returns:
        Dict with material_domain, blend_mode, shading_model
    """
    if not material or not isinstance(material, unreal.Material):
        return {}

    return {
        "material_domain": material.get_editor_property("material_domain"),
        "blend_mode": material.get_editor_property("blend_mode"),
        "shading_model": material.get_editor_property("shading_model"),
    }


def check_material_compatibility(mesh_material, master_material: unreal.Material) -> tuple:
    """Check if a mesh's material is compatible with the master material.

    Compares Material Domain, Blend Mode, and Shading Model.

    Args:
        mesh_material: The material currently on the mesh (can be Material or MaterialInstance)
        master_material: The master material to compare against

    Returns:
        Tuple of (is_compatible: bool, reason: str)
    """
    if mesh_material is None:
        return False, "No material assigned"

    # Get base material from mesh's material (in case it's an instance)
    mesh_base = get_base_material(mesh_material)
    if mesh_base is None:
        return False, "Could not find base material"

    # Get master's base material (in case user provided an instance)
    master_base = get_base_material(master_material)
    if master_base is None:
        return False, "Could not find master base material"

    # Get shading info
    mesh_info = get_material_shading_info(mesh_base)
    master_info = get_material_shading_info(master_base)

    if not mesh_info or not master_info:
        return False, "Could not read material properties"

    # Compare properties
    mismatches = []

    if mesh_info["material_domain"] != master_info["material_domain"]:
        mismatches.append(f"Domain: {mesh_info['material_domain']} vs {master_info['material_domain']}")

    if mesh_info["blend_mode"] != master_info["blend_mode"]:
        mismatches.append(f"BlendMode: {mesh_info['blend_mode']} vs {master_info['blend_mode']}")

    if mesh_info["shading_model"] != master_info["shading_model"]:
        mismatches.append(f"ShadingModel: {mesh_info['shading_model']} vs {master_info['shading_model']}")

    if mismatches:
        return False, "; ".join(mismatches)

    return True, ""


# =============================================================================
# Texture Discovery from Material Dependencies
# =============================================================================

def get_mesh_material_slots(mesh: unreal.StaticMesh) -> list:
    """Get all material slots from a static mesh with detailed info.

    Args:
        mesh: The static mesh to examine

    Returns:
        List of dicts with keys: 'index', 'slot_name', 'material'
    """
    slots = []
    static_materials = mesh.get_editor_property("static_materials")

    for i, static_mat in enumerate(static_materials):
        mat_interface = static_mat.get_editor_property("material_interface")
        slot_name = str(static_mat.get_editor_property("material_slot_name"))

        # Use index-based name if slot name is empty or "None"
        if not slot_name or slot_name == "None":
            slot_name = f"Slot{i}"

        slots.append({
            'index': i,
            'slot_name': slot_name,
            'material': mat_interface
        })

    return slots


def get_mesh_materials(mesh: unreal.StaticMesh) -> list:
    """Get all materials assigned to a static mesh."""
    slots = get_mesh_material_slots(mesh)
    return [slot['material'] for slot in slots if slot['material']]


def get_material_texture_dependencies(material) -> list:
    """Get all texture assets that a material depends on.

    Args:
        material: A Material or MaterialInstance asset

    Returns:
        List of texture assets used by the material
    """
    if not material:
        return []

    material_path = material.get_path_name()
    # Remove the asset name suffix (e.g., "/Game/Mat/M_Test.M_Test" -> "/Game/Mat/M_Test")
    if '.' in material_path:
        material_path = material_path.split('.')[0]

    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    # Get dependencies of the material
    dependencies = asset_registry.get_dependencies(
        material_path,
        unreal.AssetRegistryDependencyOptions()
    )

    textures = []
    for dep_path in dependencies:
        dep_path_str = str(dep_path)

        # Load the asset to check if it's a texture
        asset = unreal.EditorAssetLibrary.load_asset(dep_path_str)
        if asset and isinstance(asset, unreal.Texture):
            textures.append(asset)

    return textures


def get_textures_from_mesh_materials(mesh: unreal.StaticMesh) -> list:
    """Get all textures used by a mesh's materials.

    This finds textures by examining the mesh's material dependencies,
    which is more reliable than name-based matching.

    Args:
        mesh: The static mesh to examine

    Returns:
        List of unique texture assets
    """
    all_textures = []
    seen_paths = set()

    materials = get_mesh_materials(mesh)

    for material in materials:
        textures = get_material_texture_dependencies(material)
        for tex in textures:
            tex_path = tex.get_path_name()
            if tex_path not in seen_paths:
                seen_paths.add(tex_path)
                all_textures.append(tex)

    return all_textures


def find_textures_from_mesh(mesh: unreal.StaticMesh, patterns: dict = None, packing_mode: str = "STANDARD") -> dict:
    """Find and categorize textures from a mesh's existing materials.

    This is the preferred method - it finds textures by examining what
    the mesh's current materials actually use, then categorizes them
    by channel type based on naming patterns.

    Args:
        mesh: The static mesh to examine
        patterns: Texture type patterns dict
        packing_mode: "STANDARD" for separate textures, "ORM" for packed ORM textures

    Returns:
        Dict mapping texture types to texture assets
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    textures = get_textures_from_mesh_materials(mesh)

    if not textures:
        return {}

    unreal.log(f"  Found {len(textures)} textures from mesh materials:")

    matched_textures = {}
    include_orm = (packing_mode == "ORM")

    for texture in textures:
        tex_name = texture.get_name()
        tex_type = match_texture_to_type(tex_name, patterns, include_orm=include_orm)

        if tex_type:
            if tex_type not in matched_textures:
                matched_textures[tex_type] = texture
                unreal.log(f"    [{tex_type}] {tex_name}")
            else:
                unreal.log(f"    [{tex_type}] {tex_name} (duplicate, skipped)")
        else:
            unreal.log(f"    [Unknown] {tex_name}")

    return matched_textures


# =============================================================================
# ORM/ARM Packing Detection
# =============================================================================

def detect_texture_packing_mode(textures: list) -> str:
    """Detect if textures use ORM/ARM packing or separate channels.

    Examines texture names to determine which packing mode is used.

    Args:
        textures: List of texture assets to examine

    Returns:
        "ORM" if ORM/ARM packed textures detected (compatible)
        "RMA" if RMA packed textures detected (incompatible - will skip)
        "STANDARD" if separate channel textures (no packed texture found)
    """
    for texture in textures:
        tex_name = texture.get_name().lower()

        # Check for ORM/ARM patterns first (compatible)
        for pattern in ORM_PATTERNS:
            if pattern.lower() in tex_name:
                return "ORM"

        # Check for RMA patterns (incompatible)
        for pattern in RMA_PATTERNS:
            if pattern.lower() in tex_name:
                return "RMA"

    return "STANDARD"


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_content_path(path: str) -> str:
    """Normalize a content path to ensure it starts with /Game."""
    if not path:
        return path
    if not path.startswith("/"):
        path = "/" + path
    if not path.startswith("/Game"):
        path = "/Game" + path
    return path


def ensure_folder_exists(folder_path: str) -> bool:
    """Create a content folder if it doesn't exist.

    Args:
        folder_path: Content path like "/Game/Materials/Instances"

    Returns:
        True if folder exists or was successfully created
    """
    folder_path = normalize_content_path(folder_path)

    if unreal.EditorAssetLibrary.does_directory_exist(folder_path):
        return True

    # Create the folder
    success = unreal.EditorAssetLibrary.make_directory(folder_path)
    if success:
        unreal.log(f"  Created folder: {folder_path}")
    else:
        unreal.log_error(f"  Failed to create folder: {folder_path}")

    return success


# =============================================================================
# Core Functions
# =============================================================================

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
    """Get all static mesh assets from a content folder path."""
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    if not folder_path.startswith("/"):
        folder_path = "/" + folder_path
    if not folder_path.startswith("/Game"):
        folder_path = "/Game" + folder_path

    asset_data_list = asset_registry.get_assets_by_path(folder_path, recursive=recursive)

    static_meshes = []
    for asset_data in asset_data_list:
        if asset_data.asset_class_path.asset_name == "StaticMesh":
            asset = asset_data.get_asset()
            if asset:
                static_meshes.append(asset)

    return static_meshes


def find_textures_in_folder(folder_path: str, recursive: bool = True) -> list:
    """Find all texture assets in a folder."""
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    if not folder_path.startswith("/"):
        folder_path = "/" + folder_path
    if not folder_path.startswith("/Game"):
        folder_path = "/Game" + folder_path

    asset_data_list = asset_registry.get_assets_by_path(folder_path, recursive=recursive)

    textures = []
    for asset_data in asset_data_list:
        class_name = str(asset_data.asset_class_path.asset_name)
        if "Texture" in class_name:
            asset = asset_data.get_asset()
            if asset:
                textures.append(asset)

    return textures


def match_texture_to_type(texture_name: str, patterns: dict = None, include_orm: bool = False) -> Optional[str]:
    """Match a texture name to a texture type based on patterns.

    Args:
        texture_name: Name of the texture to match
        patterns: Pattern dict to use (default: DEFAULT_PATTERNS)
        include_orm: If True, also check for ORM packed texture patterns

    Returns:
        Texture type string (e.g., "BaseColor", "Normal", "ORM") or None
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    texture_name_lower = texture_name.lower()

    # Check for ORM patterns first if enabled (before checking individual channels)
    if include_orm:
        for pattern in ORM_PATTERNS:
            if pattern.lower() in texture_name_lower:
                return "ORM"

    for tex_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if pattern.lower() in texture_name_lower:
                return tex_type

    return None


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_texture_base_name(texture_name: str, patterns: dict = None) -> str:
    """Extract the base name from a texture, removing prefix and type suffix.

    Handles textures like:
        T_MPT_01_Cart_02_Normal -> MPT_01_Cart_02
        T_MPT_01_Cart_02_Normal_01 -> MPT_01_Cart_02
        T_Chair_BaseColor -> Chair
    """
    import re

    if patterns is None:
        patterns = DEFAULT_PATTERNS

    # Remove common prefixes
    base = texture_name
    for prefix in ["T_", "Tex_", "TX_"]:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    # Remove trailing numeric suffix first (e.g., _01, _02, _1, _2)
    # This handles cases like T_Name_Normal_01
    base = re.sub(r'_\d+$', '', base)

    # Remove type suffixes - find and remove the matching suffix
    base_lower = base.lower()
    for tex_type, pattern_list in patterns.items():
        found_suffix = False
        for pattern in pattern_list:
            pattern_lower = pattern.lower()
            if base_lower.endswith(pattern_lower):
                base = base[:len(base) - len(pattern)]
                found_suffix = True
                break
        if found_suffix:
            break

    # Remove any remaining trailing numeric suffix after channel removal
    base = re.sub(r'_\d+$', '', base)

    return base


def get_mesh_base_name(mesh_name: str) -> str:
    """Extract the base name from a mesh, removing common prefixes."""
    base = mesh_name
    for prefix in ["SM_", "S_", "Mesh_", "M_"]:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    return base


def find_textures_for_mesh(mesh_name: str, textures: list, patterns: dict = None, fuzzy_threshold: float = 0.6) -> tuple:
    """Find textures that match a mesh name.

    Args:
        mesh_name: Name of the static mesh
        textures: List of texture assets to search
        patterns: Texture type patterns dict
        fuzzy_threshold: Minimum similarity ratio for fuzzy matching (0.0-1.0)

    Returns:
        Tuple of (matched_textures dict, is_fuzzy_match bool)
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    # Extract mesh base name (without prefix)
    base_name = get_mesh_base_name(mesh_name)
    base_name_lower = base_name.lower()
    matched_textures = {}

    # First pass: exact matching (base name contained in texture base name)
    for texture in textures:
        tex_name = texture.get_name()
        tex_base = get_texture_base_name(tex_name, patterns)
        tex_base_lower = tex_base.lower()

        # Check if mesh base name is contained in texture base name
        if base_name_lower in tex_base_lower or tex_base_lower.startswith(base_name_lower):
            tex_type = match_texture_to_type(tex_name, patterns)
            if tex_type and tex_type not in matched_textures:
                matched_textures[tex_type] = texture

    # If exact matches found, return them
    if matched_textures:
        return matched_textures, False

    # Second pass: fuzzy matching using Levenshtein distance
    # Collect ALL candidates with their similarity scores for logging
    all_candidates = []  # list of (texture, similarity, tex_base, tex_type)

    for texture in textures:
        tex_name = texture.get_name()
        tex_base = get_texture_base_name(tex_name, patterns)
        tex_base_lower = tex_base.lower()

        # Calculate similarity
        distance = levenshtein_distance(base_name_lower, tex_base_lower)
        max_len = max(len(base_name_lower), len(tex_base_lower))
        if max_len == 0:
            continue

        similarity = 1.0 - (distance / max_len)
        tex_type = match_texture_to_type(tex_name, patterns)

        if tex_type:
            all_candidates.append((texture, similarity, tex_base, tex_type))

    # Sort by similarity (highest first)
    all_candidates.sort(key=lambda x: x[1], reverse=True)

    # Log top 3 candidates for debugging (always, even if below threshold)
    if all_candidates:
        unreal.log(f"  Fuzzy search - looking for: '{base_name}'")
        unreal.log(f"  Top 3 candidates (threshold: {fuzzy_threshold*100:.0f}%):")
        for i, (tex, sim, tex_base, tex_type) in enumerate(all_candidates[:3]):
            match_status = "MATCH" if sim >= fuzzy_threshold else "below threshold"
            unreal.log(f"    {i+1}. '{tex_base}' [{tex_type}] = {sim*100:.1f}% ({match_status})")

    # Filter by threshold and keep best match per type
    fuzzy_matches = {}  # tex_type -> (texture, similarity, tex_base_name)

    for texture, similarity, tex_base, tex_type in all_candidates:
        if similarity >= fuzzy_threshold:
            if tex_type not in fuzzy_matches or similarity > fuzzy_matches[tex_type][1]:
                fuzzy_matches[tex_type] = (texture, similarity, tex_base)

    # Convert to matched_textures format
    for tex_type, (texture, similarity, tex_base) in fuzzy_matches.items():
        matched_textures[tex_type] = texture

    return matched_textures, bool(matched_textures)


def derive_mi_name_from_material(source_material) -> str:
    """Derive a material instance name from a source material.

    Args:
        source_material: The material currently on the mesh slot (Material or MaterialInstance)

    Returns:
        MI name string (e.g., "MI_Wood" from "M_Wood" or "MI_Wood")
    """
    source_name = source_material.get_name()

    # If it's already a MaterialInstance, keep the same name
    if isinstance(source_material, unreal.MaterialInstance):
        # Already has MI_ prefix or similar, use as-is
        return source_name

    # It's a Material - convert M_ prefix to MI_
    if source_name.startswith("M_"):
        return "MI_" + source_name[2:]
    elif source_name.startswith("Mat_"):
        return "MI_" + source_name[4:]
    else:
        # No recognized prefix, just prepend MI_
        return "MI_" + source_name


def create_material_instance(
    mesh: unreal.StaticMesh,
    master_material: unreal.Material,
    source_material,
    output_folder: Optional[str] = None,
    overwrite: bool = False
) -> tuple:
    """Create a material instance based on the source material's name.

    Args:
        mesh: The static mesh (used for default output folder)
        master_material: The parent material for the new MI
        source_material: The material currently on the mesh slot (used for naming)
        output_folder: Optional folder to save MI in. If None, saves next to mesh.
                       Folder will be created if it doesn't exist.
        overwrite: If True, delete and recreate existing MI. If False, reuse existing.

    Returns:
        Tuple of (MaterialInstanceConstant or None, needs_texture_assignment: bool)
        - If existing MI reused: (mi, False) - skip texture assignment
        - If new MI created: (mi, True) - needs texture assignment
        - If failed: (None, False)
    """
    mesh_path = mesh.get_path_name()

    if output_folder:
        package_path = normalize_content_path(output_folder)
        # Ensure the output folder exists, create if needed
        if not ensure_folder_exists(package_path):
            unreal.log_error(f"  Cannot create output folder: {package_path}")
            return None, False
    else:
        # Use same folder as mesh
        package_path = "/".join(mesh_path.split("/")[:-1])

    # Derive MI name from source material
    mi_name = derive_mi_name_from_material(source_material)
    full_path = f"{package_path}/{mi_name}"

    if unreal.EditorAssetLibrary.does_asset_exist(full_path):
        if overwrite:
            unreal.log(f"  Deleting existing MI: {mi_name}")
            unreal.EditorAssetLibrary.delete_asset(full_path)
        else:
            # Reuse existing MI, skip texture assignment
            unreal.log(f"  Using existing MI: {mi_name}")
            existing_mi = unreal.EditorAssetLibrary.load_asset(full_path)
            return existing_mi, False

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    mi_factory = unreal.MaterialInstanceConstantFactoryNew()

    # Create the asset first
    material_instance = asset_tools.create_asset(
        mi_name,
        package_path,
        unreal.MaterialInstanceConstant,
        mi_factory
    )

    if material_instance:
        # Set the parent material after creation
        material_instance.set_editor_property("parent", master_material)
        unreal.log(f"  Created: {mi_name} -> {package_path}")
        return material_instance, True  # New MI, needs texture assignment
    else:
        unreal.log_error(f"  Failed to create: {mi_name}")
        return None, False


def find_texture_parameter_name(
    material_instance: unreal.MaterialInstanceConstant,
    tex_type: str,
    param_names: dict = None
) -> Optional[str]:
    """Find the actual parameter name in a material for a texture type.

    Args:
        material_instance: The material instance to search
        tex_type: Texture type (e.g., "BaseColor", "ORM")
        param_names: Parameter names dict to use (default: MATERIAL_PARAMETER_NAMES)
    """
    if param_names is None:
        param_names = MATERIAL_PARAMETER_NAMES

    parent = material_instance.get_editor_property("parent")
    if not parent:
        return None

    texture_param_infos = unreal.MaterialEditingLibrary.get_texture_parameter_names(parent)

    if tex_type not in param_names:
        return None

    possible_names = param_names[tex_type]

    for param_name in texture_param_infos:
        param_str = str(param_name)
        for possible in possible_names:
            if possible.lower() in param_str.lower():
                return param_str

    return None


def assign_textures_to_material(
    material_instance: unreal.MaterialInstanceConstant,
    textures: dict,
    param_names: dict = None
) -> int:
    """Assign textures to a material instance.

    Args:
        material_instance: The material instance to assign textures to
        textures: Dict mapping texture types to texture assets
        param_names: Parameter names dict to use (default: MATERIAL_PARAMETER_NAMES)
    """
    if param_names is None:
        param_names = MATERIAL_PARAMETER_NAMES

    assigned_count = 0

    for tex_type, texture in textures.items():
        param_name = find_texture_parameter_name(material_instance, tex_type, param_names)

        if param_name:
            success = unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
                material_instance, param_name, texture
            )
            if success:
                assigned_count += 1
                unreal.log(f"    {tex_type}: {texture.get_name()}")
        else:
            direct_names = param_names.get(tex_type, [tex_type])
            for name in direct_names:
                success = unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
                    material_instance, name, texture
                )
                if success:
                    assigned_count += 1
                    unreal.log(f"    {tex_type}: {texture.get_name()}")
                    break

    return assigned_count


def assign_material_to_mesh(mesh: unreal.StaticMesh, material_instance: unreal.MaterialInstanceConstant, slot_index: int = 0) -> bool:
    """Assign a material instance to a static mesh's material slot.

    Args:
        mesh: The static mesh to modify
        material_instance: The material instance to assign
        slot_index: Material slot index (default 0)

    Note: Does NOT save the mesh. Meshes should be saved manually from the editor
    to avoid issues with Fab/external assets being reset on reload.
    """
    static_materials = mesh.get_editor_property("static_materials")

    if slot_index >= len(static_materials):
        unreal.log_error(f"  Slot index {slot_index} out of range")
        return False

    # Mark mesh as modified BEFORE making changes (standard UE pattern)
    mesh.modify()

    # Get existing material entry and only change the material interface
    # This preserves UVChannelData and other properties
    existing_material = static_materials[slot_index]
    existing_material.set_editor_property("material_interface", material_instance)

    # Update the list
    static_materials[slot_index] = existing_material
    mesh.set_editor_property("static_materials", static_materials)

    return True


# =============================================================================
# Blueprint-Callable Functions
# =============================================================================

def process_selected(
    master_material_path: str,
    texture_folder: str,
    master_material_orm_path: Optional[str] = None,
    output_folder: Optional[str] = None,
    assign_to_mesh: bool = True,
    overwrite: bool = False
) -> int:
    """
    Process currently selected static meshes.

    Supports automatic ORM/ARM texture detection:
    - If ORM/ARM textures detected: uses master_material_orm_path (if provided)
    - If RMA textures detected: SKIPS mesh (incompatible channel order)
    - Otherwise: uses standard master_material_path

    Args:
        master_material_path: Content path to standard master material
        texture_folder: Content path to texture folder
        master_material_orm_path: Optional content path to ORM master material.
                                  If None, meshes with ORM textures will be skipped.
        output_folder: Optional folder for material instances. If None, saves next to mesh.
                       Folder will be created if it doesn't exist.
        assign_to_mesh: Whether to assign MI to mesh's material slot
        overwrite: If True, overwrite existing material instances

    Call from Blueprint:
        # Standard only:
        import material_assigner_bp as mabp; mabp.ps("/Game/Materials/M_Master", "/Game/Textures")

        # With ORM support:
        import material_assigner_bp as mabp; mabp.ps("/Game/Materials/M_Master", "/Game/Textures", "/Game/Materials/M_Master_ORM")

        # Overwrite existing MIs:
        import material_assigner_bp as mabp; mabp.ps("/Game/Materials/M_Master", "/Game/Textures", "/Game/Materials/M_Master_ORM", overwrite=True)
    """
    unreal.log("")
    unreal.log("=" * 50)
    unreal.log("MATERIAL ASSIGNER")
    unreal.log("=" * 50)

    # Load master material (standard)
    master_material = unreal.EditorAssetLibrary.load_asset(master_material_path)
    if not master_material:
        unreal.log_error(f"Master material not found: {master_material_path}")
        return 0

    # Load ORM master material if provided
    master_material_orm = None
    if master_material_orm_path:
        master_material_orm = unreal.EditorAssetLibrary.load_asset(master_material_orm_path)
        if not master_material_orm:
            unreal.log_error(f"ORM master material not found: {master_material_orm_path}")
            return 0

    unreal.log(f"Master Material: {master_material.get_name()}")
    if master_material_orm:
        unreal.log(f"ORM Material:    {master_material_orm.get_name()}")
    else:
        unreal.log(f"ORM Material:    (not provided - ORM meshes will be skipped)")
    unreal.log(f"Texture Folder: {texture_folder}")
    if output_folder:
        unreal.log(f"Output Folder:  {output_folder}")
    else:
        unreal.log(f"Output Folder:  (next to each mesh)")
    unreal.log(f"Overwrite:      {overwrite}")

    # Get selected meshes
    meshes = get_selected_static_meshes()
    if not meshes:
        unreal.log_warning("No static meshes selected!")
        return 0

    unreal.log(f"Processing {len(meshes)} mesh(es)...")
    unreal.log("-" * 50)

    # Get textures first (before creating any MIs)
    textures = find_textures_in_folder(texture_folder)
    unreal.log(f"Found {len(textures)} textures")

    # Track assets to save at end (batching for performance)
    assets_to_save = []
    num_meshes = len(meshes)
    processed = 0
    skipped = 0

    # Process each mesh with progress dialog
    with unreal.ScopedSlowTask(num_meshes, "Assigning Materials...") as slow_task:
        slow_task.make_dialog(True)  # Show dialog with cancel button

        for mesh in meshes:
            if slow_task.should_cancel():
                unreal.log_warning("Operation cancelled by user")
                break

            mesh_name = mesh.get_name()
            slow_task.enter_progress_frame(1, f"Processing: {mesh_name}")
            unreal.log(f"\n[{mesh_name}]")

            # Get all material slots
            material_slots = get_mesh_material_slots(mesh)
            if not material_slots:
                unreal.log_warning("  SKIPPED: No material slots on mesh")
                skipped += 1
                continue

            # Find matching textures by mesh name FIRST (before processing slots)
            matched, is_fuzzy = find_textures_for_mesh(mesh_name, textures)

            if not matched:
                unreal.log_warning("  No matching textures found (exact or fuzzy)")
                skipped += 1
                continue

            # Detect packing mode from matched textures
            matched_tex_list = list(matched.values())
            packing_mode = detect_texture_packing_mode(matched_tex_list)

            # Handle RMA (incompatible)
            if packing_mode == "RMA":
                unreal.log_warning("  SKIPPED: RMA packed textures detected (incompatible channel order)")
                skipped += 1
                continue

            # Select appropriate master material based on packing mode
            if packing_mode == "ORM":
                if not master_material_orm:
                    unreal.log_warning("  SKIPPED: ORM textures detected but no ORM material provided")
                    skipped += 1
                    continue
                selected_master = master_material_orm
                param_names = MATERIAL_PARAMETER_NAMES_ORM
                packing_label = "ORM"
            else:
                selected_master = master_material
                param_names = MATERIAL_PARAMETER_NAMES
                packing_label = "Standard"

            if is_fuzzy:
                unreal.log_warning("  WARNING: No exact match found - using fuzzy matching:")
                for tex_type, texture in matched.items():
                    unreal.log_warning(f"    {tex_type}: {texture.get_name()}")

            # Determine if we need slot names in MI names (multiple slots)
            use_slot_names = len(material_slots) > 1
            slots_processed = 0

            # Process each material slot
            for slot in material_slots:
                slot_index = slot['index']
                slot_name = slot['slot_name']
                slot_material = slot['material']

                slot_label = f"[Slot {slot_index}: {slot_name}]" if use_slot_names else ""

                if not slot_material:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: No material assigned")
                    else:
                        unreal.log_warning("  SKIPPED: No material assigned")
                    continue

                # Check compatibility with SELECTED master
                is_compatible, reason = check_material_compatibility(slot_material, selected_master)
                if not is_compatible:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: Incompatible - {reason}")
                    else:
                        unreal.log_warning(f"  SKIPPED: Incompatible material - {reason}")
                    continue

                # Create material instance (named after source material)
                mi, needs_assignment = create_material_instance(mesh, selected_master, slot_material, output_folder, overwrite)
                if not mi:
                    continue

                # Log slot info
                if use_slot_names:
                    unreal.log(f"  {slot_label} Packing: {packing_label}")
                else:
                    unreal.log(f"  Packing: {packing_label}")

                # Assign textures only if this is a new MI
                if needs_assignment:
                    unreal.log("  Assigning textures:")
                    assign_textures_to_material(mi, matched, param_names)
                    assets_to_save.append(mi)

                # Assign to mesh at correct slot index
                if assign_to_mesh:
                    assign_material_to_mesh(mesh, mi, slot_index)

                slots_processed += 1

            if slots_processed > 0:
                processed += 1
            else:
                skipped += 1

    # Save only new Material Instances (not meshes - save those manually from editor)
    if assets_to_save:
        unreal.log(f"\nSaving {len(assets_to_save)} new material instance(s)...")
        for asset in assets_to_save:
            unreal.EditorAssetLibrary.save_loaded_asset(asset)

    unreal.log("")
    unreal.log("=" * 50)
    unreal.log(f"COMPLETE: {processed}/{num_meshes} meshes processed, {skipped} skipped")
    unreal.log("NOTE: Static meshes are NOT auto-saved. Save them manually")
    unreal.log("      from Content Browser (Ctrl+Shift+S) to persist changes.")
    unreal.log("=" * 50)

    return processed


def process_folder(
    mesh_folder: str,
    master_material_path: str,
    texture_folder: str,
    master_material_orm_path: Optional[str] = None,
    output_folder: Optional[str] = None,
    assign_to_mesh: bool = True,
    overwrite: bool = False
) -> int:
    """
    Process all static meshes in a folder.

    Supports automatic ORM/ARM texture detection:
    - If ORM/ARM textures detected: uses master_material_orm_path (if provided)
    - If RMA textures detected: SKIPS mesh (incompatible channel order)
    - Otherwise: uses standard master_material_path

    Args:
        mesh_folder: Content path to folder containing static meshes
        master_material_path: Content path to standard master material
        texture_folder: Content path to texture folder
        master_material_orm_path: Optional content path to ORM master material.
                                  If None, meshes with ORM textures will be skipped.
        output_folder: Optional folder for material instances. If None, saves next to mesh.
                       Folder will be created if it doesn't exist.
        assign_to_mesh: Whether to assign MI to mesh's material slot
        overwrite: If True, overwrite existing material instances

    Call from Blueprint:
        # Standard only:
        import material_assigner_bp as mabp; mabp.pf("/Game/Meshes", "/Game/Materials/M_Master", "/Game/Textures")

        # With ORM support:
        import material_assigner_bp as mabp; mabp.pf("/Game/Meshes", "/Game/Materials/M_Master", "/Game/Textures", "/Game/Materials/M_Master_ORM")

        # Overwrite existing MIs:
        import material_assigner_bp as mabp; mabp.pf("/Game/Meshes", "/Game/Materials/M_Master", "/Game/Textures", "/Game/Materials/M_Master_ORM", overwrite=True)
    """
    unreal.log("")
    unreal.log("=" * 50)
    unreal.log("MATERIAL ASSIGNER")
    unreal.log("=" * 50)

    # Load master material (standard)
    master_material = unreal.EditorAssetLibrary.load_asset(master_material_path)
    if not master_material:
        unreal.log_error(f"Master material not found: {master_material_path}")
        return 0

    # Load ORM master material if provided
    master_material_orm = None
    if master_material_orm_path:
        master_material_orm = unreal.EditorAssetLibrary.load_asset(master_material_orm_path)
        if not master_material_orm:
            unreal.log_error(f"ORM master material not found: {master_material_orm_path}")
            return 0

    unreal.log(f"Mesh Folder:     {mesh_folder}")
    unreal.log(f"Master Material: {master_material.get_name()}")
    if master_material_orm:
        unreal.log(f"ORM Material:    {master_material_orm.get_name()}")
    else:
        unreal.log(f"ORM Material:    (not provided - ORM meshes will be skipped)")
    unreal.log(f"Texture Folder:  {texture_folder}")
    if output_folder:
        unreal.log(f"Output Folder:   {output_folder}")
    else:
        unreal.log(f"Output Folder:   (next to each mesh)")
    unreal.log(f"Overwrite:       {overwrite}")

    # Get meshes
    meshes = get_static_meshes_from_folder(mesh_folder)
    if not meshes:
        unreal.log_warning(f"No static meshes in: {mesh_folder}")
        return 0

    unreal.log(f"Processing {len(meshes)} mesh(es)...")
    unreal.log("-" * 50)

    # Get textures first (before creating any MIs)
    textures = find_textures_in_folder(texture_folder)
    unreal.log(f"Found {len(textures)} textures")

    # Track assets to save at end (batching for performance)
    assets_to_save = []
    num_meshes = len(meshes)
    processed = 0
    skipped = 0

    # Process each mesh with progress dialog
    with unreal.ScopedSlowTask(num_meshes, "Assigning Materials...") as slow_task:
        slow_task.make_dialog(True)

        for mesh in meshes:
            if slow_task.should_cancel():
                unreal.log_warning("Operation cancelled by user")
                break

            mesh_name = mesh.get_name()
            slow_task.enter_progress_frame(1, f"Processing: {mesh_name}")
            unreal.log(f"\n[{mesh_name}]")

            # Get all material slots
            material_slots = get_mesh_material_slots(mesh)
            if not material_slots:
                unreal.log_warning("  SKIPPED: No material slots on mesh")
                skipped += 1
                continue

            # Find matching textures by mesh name FIRST (before processing slots)
            matched, is_fuzzy = find_textures_for_mesh(mesh_name, textures)

            if not matched:
                unreal.log_warning("  No matching textures found (exact or fuzzy)")
                skipped += 1
                continue

            # Detect packing mode from matched textures
            matched_tex_list = list(matched.values())
            packing_mode = detect_texture_packing_mode(matched_tex_list)

            # Handle RMA (incompatible)
            if packing_mode == "RMA":
                unreal.log_warning("  SKIPPED: RMA packed textures detected (incompatible channel order)")
                skipped += 1
                continue

            # Select appropriate master material based on packing mode
            if packing_mode == "ORM":
                if not master_material_orm:
                    unreal.log_warning("  SKIPPED: ORM textures detected but no ORM material provided")
                    skipped += 1
                    continue
                selected_master = master_material_orm
                param_names = MATERIAL_PARAMETER_NAMES_ORM
                packing_label = "ORM"
            else:
                selected_master = master_material
                param_names = MATERIAL_PARAMETER_NAMES
                packing_label = "Standard"

            if is_fuzzy:
                unreal.log_warning("  WARNING: No exact match found - using fuzzy matching:")
                for tex_type, texture in matched.items():
                    unreal.log_warning(f"    {tex_type}: {texture.get_name()}")

            # Determine if we need slot names in MI names (multiple slots)
            use_slot_names = len(material_slots) > 1
            slots_processed = 0

            # Process each material slot
            for slot in material_slots:
                slot_index = slot['index']
                slot_name = slot['slot_name']
                slot_material = slot['material']

                slot_label = f"[Slot {slot_index}: {slot_name}]" if use_slot_names else ""

                if not slot_material:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: No material assigned")
                    else:
                        unreal.log_warning("  SKIPPED: No material assigned")
                    continue

                # Check compatibility with SELECTED master
                is_compatible, reason = check_material_compatibility(slot_material, selected_master)
                if not is_compatible:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: Incompatible - {reason}")
                    else:
                        unreal.log_warning(f"  SKIPPED: Incompatible material - {reason}")
                    continue

                # Create material instance (named after source material)
                mi, needs_assignment = create_material_instance(mesh, selected_master, slot_material, output_folder, overwrite)
                if not mi:
                    continue

                # Log slot info
                if use_slot_names:
                    unreal.log(f"  {slot_label} Packing: {packing_label}")
                else:
                    unreal.log(f"  Packing: {packing_label}")

                # Assign textures only if this is a new MI
                if needs_assignment:
                    unreal.log("  Assigning textures:")
                    assign_textures_to_material(mi, matched, param_names)
                    assets_to_save.append(mi)

                # Assign to mesh at correct slot index
                if assign_to_mesh:
                    assign_material_to_mesh(mesh, mi, slot_index)

                slots_processed += 1

            if slots_processed > 0:
                processed += 1
            else:
                skipped += 1

    # Save only new Material Instances (not meshes - save those manually from editor)
    if assets_to_save:
        unreal.log(f"\nSaving {len(assets_to_save)} new material instance(s)...")
        for asset in assets_to_save:
            unreal.EditorAssetLibrary.save_loaded_asset(asset)

    unreal.log("")
    unreal.log("=" * 50)
    unreal.log(f"COMPLETE: {processed}/{num_meshes} meshes processed, {skipped} skipped")
    unreal.log("NOTE: Static meshes are NOT auto-saved. Save them manually")
    unreal.log("      from Content Browser (Ctrl+Shift+S) to persist changes.")
    unreal.log("=" * 50)

    return processed


# =============================================================================
# Auto-Discovery Functions (Recommended - Uses Material Dependencies)
# =============================================================================

def process_selected_auto(
    master_material_path: str,
    master_material_orm_path: Optional[str] = None,
    output_folder: Optional[str] = None,
    assign_to_mesh: bool = True,
    overwrite: bool = False
) -> int:
    """
    Process selected meshes by discovering textures from their existing materials.

    This is the RECOMMENDED method - it finds textures by examining what
    the mesh's current materials actually use, rather than searching by name.
    No texture folder needed!

    Supports automatic ORM/ARM texture detection:
    - If ORM/ARM textures detected: uses master_material_orm_path (if provided)
    - If RMA textures detected: SKIPS mesh (incompatible channel order)
    - Otherwise: uses standard master_material_path

    Args:
        master_material_path: Content path to standard master material
        master_material_orm_path: Optional content path to ORM master material.
                                  If None, meshes with ORM textures will be skipped.
        output_folder: Optional folder for material instances. If None, saves next to mesh.
        assign_to_mesh: Whether to assign MI to mesh's material slot
        overwrite: If True, overwrite existing material instances

    Call from Blueprint:
        # Standard only:
        import material_assigner_bp as mabp; mabp.psa("/Game/Materials/M_Master")

        # With ORM support:
        import material_assigner_bp as mabp; mabp.psa("/Game/Materials/M_Master", "/Game/Materials/M_Master_ORM")

        # Overwrite existing MIs:
        import material_assigner_bp as mabp; mabp.psa("/Game/Materials/M_Master", "/Game/Materials/M_Master_ORM", overwrite=True)
    """
    unreal.log("")
    unreal.log("=" * 50)
    unreal.log("MATERIAL ASSIGNER (Auto-Discovery Mode)")
    unreal.log("=" * 50)

    # Load master material (standard)
    master_material = unreal.EditorAssetLibrary.load_asset(master_material_path)
    if not master_material:
        unreal.log_error(f"Master material not found: {master_material_path}")
        return 0

    # Load ORM master material if provided
    master_material_orm = None
    if master_material_orm_path:
        master_material_orm = unreal.EditorAssetLibrary.load_asset(master_material_orm_path)
        if not master_material_orm:
            unreal.log_error(f"ORM master material not found: {master_material_orm_path}")
            return 0

    unreal.log(f"Master Material: {master_material.get_name()}")
    if master_material_orm:
        unreal.log(f"ORM Material:    {master_material_orm.get_name()}")
    else:
        unreal.log(f"ORM Material:    (not provided - ORM meshes will be skipped)")
    unreal.log(f"Mode: Auto-discover textures from mesh materials")
    if output_folder:
        unreal.log(f"Output Folder:  {output_folder}")
    else:
        unreal.log(f"Output Folder:  (next to each mesh)")
    unreal.log(f"Overwrite:      {overwrite}")

    # Get selected meshes
    meshes = get_selected_static_meshes()
    if not meshes:
        unreal.log_warning("No static meshes selected!")
        return 0

    unreal.log(f"Processing {len(meshes)} mesh(es)...")
    unreal.log("-" * 50)

    # Track assets to save at end
    assets_to_save = []
    num_meshes = len(meshes)
    processed = 0
    skipped = 0

    with unreal.ScopedSlowTask(num_meshes, "Assigning Materials (Auto)...") as slow_task:
        slow_task.make_dialog(True)

        for mesh in meshes:
            if slow_task.should_cancel():
                unreal.log_warning("Operation cancelled by user")
                break

            mesh_name = mesh.get_name()
            slow_task.enter_progress_frame(1, f"Processing: {mesh_name}")
            unreal.log(f"\n[{mesh_name}]")

            # Get all material slots
            material_slots = get_mesh_material_slots(mesh)
            if not material_slots:
                unreal.log_warning("  SKIPPED: No material slots on mesh")
                skipped += 1
                continue

            # Determine if we need slot names in MI names (multiple slots)
            use_slot_names = len(material_slots) > 1
            slots_processed = 0

            # Process each material slot
            for slot in material_slots:
                slot_index = slot['index']
                slot_name = slot['slot_name']
                slot_material = slot['material']

                slot_label = f"[Slot {slot_index}: {slot_name}]" if use_slot_names else ""

                if not slot_material:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: No material assigned")
                    else:
                        unreal.log_warning("  SKIPPED: No material assigned")
                    continue

                # Get textures from this slot's material
                raw_textures = get_material_texture_dependencies(slot_material)
                if not raw_textures:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: No textures found")
                    else:
                        unreal.log_warning("  SKIPPED: No textures found in material")
                    continue

                # Detect texture packing mode for this slot
                packing_mode = detect_texture_packing_mode(raw_textures)

                # Handle RMA (incompatible)
                if packing_mode == "RMA":
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: RMA packed textures (incompatible)")
                    else:
                        unreal.log_warning("  SKIPPED: RMA packed textures detected (incompatible channel order)")
                    continue

                # Select appropriate master material based on packing mode
                if packing_mode == "ORM":
                    if not master_material_orm:
                        if use_slot_names:
                            unreal.log_warning(f"  {slot_label} SKIPPED: ORM textures but no ORM material provided")
                        else:
                            unreal.log_warning("  SKIPPED: ORM textures detected but no ORM material provided")
                        continue
                    selected_master = master_material_orm
                    param_names = MATERIAL_PARAMETER_NAMES_ORM
                    packing_label = "ORM"
                else:
                    selected_master = master_material
                    param_names = MATERIAL_PARAMETER_NAMES
                    packing_label = "Standard"

                # Check compatibility with SELECTED master
                is_compatible, reason = check_material_compatibility(slot_material, selected_master)
                if not is_compatible:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: Incompatible - {reason}")
                    else:
                        unreal.log_warning(f"  SKIPPED: Incompatible material - {reason}")
                    continue

                # Categorize textures with ORM awareness
                include_orm = (packing_mode == "ORM")
                matched = {}
                for texture in raw_textures:
                    tex_name = texture.get_name()
                    tex_type = match_texture_to_type(tex_name, include_orm=include_orm)
                    if tex_type and tex_type not in matched:
                        matched[tex_type] = texture

                if not matched:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: No matching texture types found")
                    else:
                        unreal.log_warning("  SKIPPED: No matching texture types found")
                    continue

                # Create material instance (named after source material)
                mi, needs_assignment = create_material_instance(mesh, selected_master, slot_material, output_folder, overwrite)
                if not mi:
                    continue

                # Log slot info
                if use_slot_names:
                    unreal.log(f"  {slot_label} Packing: {packing_label}")
                else:
                    unreal.log(f"  Packing: {packing_label}")

                # Assign textures only if this is a new MI
                if needs_assignment:
                    unreal.log("  Assigning textures:")
                    assign_textures_to_material(mi, matched, param_names)
                    assets_to_save.append(mi)

                # Assign to mesh at correct slot index
                if assign_to_mesh:
                    assign_material_to_mesh(mesh, mi, slot_index)

                slots_processed += 1

            if slots_processed > 0:
                processed += 1
            else:
                skipped += 1

    # Save only new Material Instances (not meshes - save those manually from editor)
    if assets_to_save:
        unreal.log(f"\nSaving {len(assets_to_save)} new material instance(s)...")
        for asset in assets_to_save:
            unreal.EditorAssetLibrary.save_loaded_asset(asset)

    unreal.log("")
    unreal.log("=" * 50)
    unreal.log(f"COMPLETE: {processed}/{num_meshes} meshes processed, {skipped} skipped")
    unreal.log("NOTE: Static meshes are NOT auto-saved. Save them manually")
    unreal.log("      from Content Browser (Ctrl+Shift+S) to persist changes.")
    unreal.log("=" * 50)

    return processed


def process_folder_auto(
    mesh_folder: str,
    master_material_path: str,
    master_material_orm_path: Optional[str] = None,
    output_folder: Optional[str] = None,
    assign_to_mesh: bool = True,
    overwrite: bool = False
) -> int:
    """
    Process all meshes in a folder by discovering textures from their existing materials.

    This is the RECOMMENDED method - no texture folder needed!

    Supports automatic ORM/ARM texture detection:
    - If ORM/ARM textures detected: uses master_material_orm_path (if provided)
    - If RMA textures detected: SKIPS mesh (incompatible channel order)
    - Otherwise: uses standard master_material_path

    Args:
        mesh_folder: Content path to folder containing static meshes
        master_material_path: Content path to standard master material
        master_material_orm_path: Optional content path to ORM master material.
                                  If None, meshes with ORM textures will be skipped.
        output_folder: Optional folder for material instances
        assign_to_mesh: Whether to assign MI to mesh's material slot
        overwrite: If True, overwrite existing material instances

    Call from Blueprint:
        # Standard only:
        import material_assigner_bp as mabp; mabp.pfa("/Game/Meshes", "/Game/Materials/M_Master")

        # With ORM support:
        import material_assigner_bp as mabp; mabp.pfa("/Game/Meshes", "/Game/Materials/M_Master", "/Game/Materials/M_Master_ORM")

        # Overwrite existing MIs:
        import material_assigner_bp as mabp; mabp.pfa("/Game/Meshes", "/Game/Materials/M_Master", "/Game/Materials/M_Master_ORM", overwrite=True)
    """
    unreal.log("")
    unreal.log("=" * 50)
    unreal.log("MATERIAL ASSIGNER (Auto-Discovery Mode)")
    unreal.log("=" * 50)

    # Load master material (standard)
    master_material = unreal.EditorAssetLibrary.load_asset(master_material_path)
    if not master_material:
        unreal.log_error(f"Master material not found: {master_material_path}")
        return 0

    # Load ORM master material if provided
    master_material_orm = None
    if master_material_orm_path:
        master_material_orm = unreal.EditorAssetLibrary.load_asset(master_material_orm_path)
        if not master_material_orm:
            unreal.log_error(f"ORM master material not found: {master_material_orm_path}")
            return 0

    unreal.log(f"Mesh Folder:     {mesh_folder}")
    unreal.log(f"Master Material: {master_material.get_name()}")
    if master_material_orm:
        unreal.log(f"ORM Material:    {master_material_orm.get_name()}")
    else:
        unreal.log(f"ORM Material:    (not provided - ORM meshes will be skipped)")
    unreal.log(f"Mode: Auto-discover textures from mesh materials")
    if output_folder:
        unreal.log(f"Output Folder:   {output_folder}")
    else:
        unreal.log(f"Output Folder:   (next to each mesh)")
    unreal.log(f"Overwrite:       {overwrite}")

    # Get meshes
    meshes = get_static_meshes_from_folder(mesh_folder)
    if not meshes:
        unreal.log_warning(f"No static meshes in: {mesh_folder}")
        return 0

    unreal.log(f"Processing {len(meshes)} mesh(es)...")
    unreal.log("-" * 50)

    assets_to_save = []
    num_meshes = len(meshes)
    processed = 0
    skipped = 0

    with unreal.ScopedSlowTask(num_meshes, "Assigning Materials (Auto)...") as slow_task:
        slow_task.make_dialog(True)

        for mesh in meshes:
            if slow_task.should_cancel():
                unreal.log_warning("Operation cancelled by user")
                break

            mesh_name = mesh.get_name()
            slow_task.enter_progress_frame(1, f"Processing: {mesh_name}")
            unreal.log(f"\n[{mesh_name}]")

            # Get all material slots
            material_slots = get_mesh_material_slots(mesh)
            if not material_slots:
                unreal.log_warning("  SKIPPED: No material slots on mesh")
                skipped += 1
                continue

            # Determine if we need slot names in MI names (multiple slots)
            use_slot_names = len(material_slots) > 1
            slots_processed = 0

            # Process each material slot
            for slot in material_slots:
                slot_index = slot['index']
                slot_name = slot['slot_name']
                slot_material = slot['material']

                slot_label = f"[Slot {slot_index}: {slot_name}]" if use_slot_names else ""

                if not slot_material:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: No material assigned")
                    else:
                        unreal.log_warning("  SKIPPED: No material assigned")
                    continue

                # Get textures from this slot's material
                raw_textures = get_material_texture_dependencies(slot_material)
                if not raw_textures:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: No textures found")
                    else:
                        unreal.log_warning("  SKIPPED: No textures found in material")
                    continue

                # Detect texture packing mode for this slot
                packing_mode = detect_texture_packing_mode(raw_textures)

                # Handle RMA (incompatible)
                if packing_mode == "RMA":
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: RMA packed textures (incompatible)")
                    else:
                        unreal.log_warning("  SKIPPED: RMA packed textures detected (incompatible channel order)")
                    continue

                # Select appropriate master material based on packing mode
                if packing_mode == "ORM":
                    if not master_material_orm:
                        if use_slot_names:
                            unreal.log_warning(f"  {slot_label} SKIPPED: ORM textures but no ORM material provided")
                        else:
                            unreal.log_warning("  SKIPPED: ORM textures detected but no ORM material provided")
                        continue
                    selected_master = master_material_orm
                    param_names = MATERIAL_PARAMETER_NAMES_ORM
                    packing_label = "ORM"
                else:
                    selected_master = master_material
                    param_names = MATERIAL_PARAMETER_NAMES
                    packing_label = "Standard"

                # Check compatibility with SELECTED master
                is_compatible, reason = check_material_compatibility(slot_material, selected_master)
                if not is_compatible:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: Incompatible - {reason}")
                    else:
                        unreal.log_warning(f"  SKIPPED: Incompatible material - {reason}")
                    continue

                # Categorize textures with ORM awareness
                include_orm = (packing_mode == "ORM")
                matched = {}
                for texture in raw_textures:
                    tex_name = texture.get_name()
                    tex_type = match_texture_to_type(tex_name, include_orm=include_orm)
                    if tex_type and tex_type not in matched:
                        matched[tex_type] = texture

                if not matched:
                    if use_slot_names:
                        unreal.log_warning(f"  {slot_label} SKIPPED: No matching texture types found")
                    else:
                        unreal.log_warning("  SKIPPED: No matching texture types found")
                    continue

                # Create material instance (named after source material)
                mi, needs_assignment = create_material_instance(mesh, selected_master, slot_material, output_folder, overwrite)
                if not mi:
                    continue

                # Log slot info
                if use_slot_names:
                    unreal.log(f"  {slot_label} Packing: {packing_label}")
                else:
                    unreal.log(f"  Packing: {packing_label}")

                # Assign textures only if this is a new MI
                if needs_assignment:
                    unreal.log("  Assigning textures:")
                    assign_textures_to_material(mi, matched, param_names)
                    assets_to_save.append(mi)

                # Assign to mesh at correct slot index
                if assign_to_mesh:
                    assign_material_to_mesh(mesh, mi, slot_index)

                slots_processed += 1

            if slots_processed > 0:
                processed += 1
            else:
                skipped += 1

    # Save only new Material Instances (not meshes - save those manually from editor)
    if assets_to_save:
        unreal.log(f"\nSaving {len(assets_to_save)} new material instance(s)...")
        for asset in assets_to_save:
            unreal.EditorAssetLibrary.save_loaded_asset(asset)

    unreal.log("")
    unreal.log("=" * 50)
    unreal.log(f"COMPLETE: {processed}/{num_meshes} meshes processed, {skipped} skipped")
    unreal.log("NOTE: Static meshes are NOT auto-saved. Save them manually")
    unreal.log("      from Content Browser (Ctrl+Shift+S) to persist changes.")
    unreal.log("=" * 50)

    return processed


# Shorthand aliases for Blueprint
ps = process_selected
pf = process_folder
psa = process_selected_auto
pfa = process_folder_auto
