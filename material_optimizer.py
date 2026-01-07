# Copyright (c) 2026-Present Nehon (github.com/Nehon)
"""
Material Optimizer - Blueprint Function Library

A Python-based Unreal Engine tool that creates optimized Material Instances from
static mesh materials. Exposed to Blueprints via @unreal.uclass() and @unreal.ufunction()
decorators, providing native Blueprint nodes under the "Material Optimizer" category.

Key Features:
- Analyzes static mesh material slots and their texture dependencies
- Creates Material Instances parented to a master material
- Automatically assigns textures to the correct material parameters
- Supports ORM/ARM packed texture workflows
- Confidence-based texture matching with mesh/material/slot name heuristics
- Two-phase workflow: analyze first, then process with user validation

Texture Packing Support:
- ORM/ARM textures: Occlusion(R), Roughness(G), Metallic(B) - SUPPORTED
- RMA textures: Roughness(R), Metallic(G), AO(B) - SKIPPED (incompatible channel order)
- RAM textures: Roughness(R), AO(G), Metallic(B) - SKIPPED (incompatible channel order)

Blueprint Usage:
    The BPMaterialOptimizer class exposes two Blueprint-callable functions:

    1. "Analyze Selected Meshes" node:
       - Analyzes Content Browser selected meshes
       - Returns JSON with texture candidates, conflicts, and confidence scores
       - No changes made to assets (read-only analysis)

    2. "Optimize Materials From Analysis" node:
       - Takes analysis JSON (potentially modified by user to resolve conflicts)
       - Creates Material Instances and assigns textures
       - Assigns new MIs to mesh material slots

Two-Phase Workflow:
    Phase 1 - Analysis:
        Call "Analyze Selected Meshes" to get a JSON report of all meshes,
        their material slots, detected textures, and any conflicts.

    Phase 2 - Processing:
        Pass the analysis JSON to "Optimize Materials From Analysis".
        The JSON can be modified between phases to:
        - Set 'selected': true on preferred texture candidates
        - Set 'skip': true on meshes to exclude
        - Set channel values to true in 'skipped_channels' to ignore specific texture types

Output:
    Results are logged to the Output Log and returned as JSON strings.
"""

import unreal
import json
from typing import Optional, List, Dict, Any


# =============================================================================
# Default texture patterns
# =============================================================================

DEFAULT_PATTERNS = {
    "BaseColor": ["_BaseColor", "_Diffuse", "_Albedo", "_Color", "_Base_Color","_BC", "_D", "_diff"],
    "Normal": ["_Normal", "_Nrm", "_N", "_NormalMap", "_norm"],
    "Roughness": ["_Roughness", "_Rough", "_R", "_roughness"],
    "Metallic": ["_Metallic", "_Metal", "_M", "_metallic", "_Metalness"],
    "AO": ["_AO", "_Occlusion", "_AmbientOcclusion", "_O", "_ao"],
    "Emissive": ["_Emissive", "_Emission", "_E", "_Glow", "_emissive"]
}

# Patterns for ORM/ARM packed textures (COMPATIBLE - same channel order: Occlusion/AO(R), Roughness(G), Metallic(B))
ORM_PATTERNS = ["_ORM", "_ARM", "_OcclusionRoughnessMetallic", "_AORoughnessMetallic"]

# Patterns for RMA packed textures (INCOMPATIBLE - different channel order: Roughness(R), Metallic(G), AO(B))
RMA_PATTERNS = ["_RMA", "_RoughnessMetallicAO", "_RoughnessMetalAO"]

# Patterns for RAM packed textures (INCOMPATIBLE - different channel order: Roughness(R), AO(B),  Metallic(G),)
RAM_PATTERNS = ["_RAM", "_RoughnessAOMetallic", "_RoughnessAOMetal", "-RAM", "RAM_"]

MATERIAL_PARAMETER_NAMES = {
    "BaseColor": ["BaseColor",  "BaseColorMap", "Base Color", "Diffuse", "Albedo", "Color"],
    "Normal": ["Normal", "NormalMap", "Normal Map"],
    "Roughness": ["Roughness","RoughnessMap", "Rough"],
    "Metallic": ["Metallic", "Matalness", "MetallicMap", "MetalnessMap", "Metal"],
    "AO": ["AO", "AmbientOcclusion", "OcclusionMap", "Ambient Occlusion", "Occlusion"],
    "Emissive": ["Emissive", "Emission", "EmissiveMap", "EmissionMap", "EmissiveColor"]
}

# Parameter names for ORM master material
MATERIAL_PARAMETER_NAMES_ORM = {
    "BaseColor": ["BaseColor",  "BaseColorMap", "Base Color", "Diffuse", "Albedo", "Color"],
    "Normal": ["Normal", "NormalMap", "Normal Map"],
    "ORM": ["ORM Map", "ORM", "ARM", "OcclusionRoughnessMetallic", "PackedTexture"],
    "Emissive": ["Emissive", "Emission", "EmissiveMap", "EmissionMap", "EmissiveColor"]
}

# =============================================================================
# Material Compatibility Checking
# =============================================================================

@unreal.uclass()
class BPMaterialOptimizer(unreal.BlueprintFunctionLibrary):
    @staticmethod
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
            return BPMaterialOptimizer.get_base_material(parent)
    
        return None

    @staticmethod
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

    @staticmethod
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
        mesh_base = BPMaterialOptimizer.get_base_material(mesh_material)
        if mesh_base is None:
            return False, "Could not find base material"
    
        # Get master's base material (in case user provided an instance)
        master_base = BPMaterialOptimizer.get_base_material(master_material)
        if master_base is None:
            return False, "Could not find master base material"
    
        # Get shading info
        mesh_info = BPMaterialOptimizer.get_material_shading_info(mesh_base)
        master_info = BPMaterialOptimizer.get_material_shading_info(master_base)
    
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
    @staticmethod
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

    @staticmethod
    def get_mesh_materials(mesh: unreal.StaticMesh) -> list:
        """Get all materials assigned to a static mesh."""
        slots = BPMaterialOptimizer.get_mesh_material_slots(mesh)
        return [slot['material'] for slot in slots if slot['material']]

    @staticmethod
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

    @staticmethod
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
    
        materials = BPMaterialOptimizer.get_mesh_materials(mesh)
    
        for material in materials:
            textures = BPMaterialOptimizer.get_material_texture_dependencies(material)
            for tex in textures:
                tex_path = tex.get_path_name()
                if tex_path not in seen_paths:
                    seen_paths.add(tex_path)
                    all_textures.append(tex)
    
        return all_textures

    @staticmethod
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
    
        textures = BPMaterialOptimizer.get_textures_from_mesh_materials(mesh)
    
        if not textures:
            return {}
    
        unreal.log(f"  Found {len(textures)} textures from mesh materials:")
    
        matched_textures = {}
        include_orm = (packing_mode == "ORM")
    
        for texture in textures:
            tex_name = texture.get_name()
            tex_type = BPMaterialOptimizer.match_texture_to_type(tex_name, patterns, include_orm=include_orm)
    
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
    @staticmethod
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
    
            for pattern in RAM_PATTERNS:
                if pattern.lower() in tex_name:
                    return "RAM"
    
        return "STANDARD"
    
    
    # =============================================================================
    # Texture Conflict Analysis (Two-Phase Workflow)
    # =============================================================================
    @staticmethod
    def get_all_texture_candidates(
        textures: list,
        patterns: dict = None,
        include_orm: bool = False,
        context_names: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize all textures, keeping ALL candidates per type.
    
        Unlike match_texture_to_type which returns only the first match,
        this function collects ALL textures that match each type.
    
        Confidence scoring:
        - Base confidence: 1.0 if only one texture matches the channel, else 1.0 / num_matches
        - Name bonus: +0.3 if texture name contains any of the context names (mesh/material/slot)
        - Results sorted by confidence descending
        - Auto-select the highest confidence texture if >= 1.0
    
        Args:
            textures: List of texture assets to categorize
            patterns: Texture type patterns dict (default: DEFAULT_PATTERNS)
            include_orm: If True, also check for ORM packed texture patterns
            context_names: Optional list of context names (mesh name, material name, slot name)
                           used for confidence bonus when texture name contains them
    
        Returns:
            Dict mapping texture types to lists of candidate dicts:
            {
                'BaseColor': [
                    {'name': 'T_Wood_BaseColor', 'path': '/Game/...', 'confidence': 1.0, 'selected': True},
                    {'name': 'T_Moss_BaseColor', 'path': '/Game/...', 'confidence': 0.8, 'selected': False}
                ],
                ...
            }
        """
        if patterns is None:
            patterns = DEFAULT_PATTERNS
    
        # Normalize context names for case-insensitive matching
        normalized_context = []
        if context_names:
            for name in context_names:
                if name:
                    # Also try without common prefixes
                    name_lower = name.lower()
                    normalized_context.append(name_lower)
                    # Remove common prefixes for better matching
                    for prefix in ["sm_", "s_", "m_", "mi_", "mat_", "t_", "tex_"]:
                        if name_lower.startswith(prefix):
                            normalized_context.append(name_lower[len(prefix):])
                            break
    
        candidates: Dict[str, List[Dict[str, Any]]] = {}
    
        # First pass: collect all matching textures per type
        for texture in textures:
            tex_name = texture.get_name()
            tex_path = texture.get_path_name()
            if '.' in tex_path:
                tex_path = tex_path.split('.')[0]
    
            tex_name_lower = tex_name.lower()
    
            # Check for ORM patterns first if enabled
            if include_orm:
                for pattern in ORM_PATTERNS:
                    if pattern.lower() in tex_name_lower:
                        if 'ORM' not in candidates:
                            candidates['ORM'] = []
                        candidates['ORM'].append({
                            'name': tex_name,
                            'path': tex_path,
                            'confidence': 0.0,
                            'selected': False
                        })
                        break
                else:
                    # No ORM pattern matched, check standard patterns
                    for tex_type, pattern_list in patterns.items():
                        for pattern in pattern_list:
                            if pattern.lower() in tex_name_lower:
                                if tex_type not in candidates:
                                    candidates[tex_type] = []
                                candidates[tex_type].append({
                                    'name': tex_name,
                                    'path': tex_path,
                                    'confidence': 0.0,
                                    'selected': False
                                })
                                break
                        else:
                            # Inner loop completed without match, try next tex_type
                            continue
                        # Inner loop broke (pattern matched), stop checking other types
                        break
            else:
                # Standard patterns only
                for tex_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        if pattern.lower() in tex_name_lower:
                            if tex_type not in candidates:
                                candidates[tex_type] = []
                            candidates[tex_type].append({
                                'name': tex_name,
                                'path': tex_path,
                                'confidence': 0.0,
                                'selected': False
                            })
                            break
                    else:
                        # Inner loop completed without match, try next tex_type
                        continue
                    # Inner loop broke (pattern matched), stop checking other types
                    break
    
        # Second pass: calculate confidence scores
        for tex_type, tex_list in candidates.items():
            num_matches = len(tex_list)
    
            for tex_entry in tex_list:
                # Base confidence: 1.0 if single match, else 1/N
                base_confidence = 1.0 if num_matches == 1 else 1.0 / num_matches
    
                # Name bonus: +0.3 if texture name contains any context name
                name_bonus = 0.0
                if normalized_context:
                    tex_name_lower = tex_entry['name'].lower()
                    for ctx_name in normalized_context:
                        if ctx_name in tex_name_lower:
                            name_bonus = 0.3
                            break
    
                tex_entry['confidence'] = base_confidence + name_bonus
    
            # Sort by confidence descending
            tex_list.sort(key=lambda x: x['confidence'], reverse=True)
    
            # Auto-select the first (highest confidence) if >= 1.0
            if tex_list and tex_list[0]['confidence'] >= 1.0:
                tex_list[0]['selected'] = True
    
        return candidates

    @staticmethod
    def _analyze_slot(
        slot_material,
        master_material,
        master_material_orm,
        slot_index: int,
        slot_name: str,
        mesh_name: str
    ) -> Dict[str, Any]:
        """Analyze a single material slot for texture conflicts.
    
        Args:
            slot_material: The material in this slot
            master_material: Standard master material
            master_material_orm: ORM master material (optional)
            slot_index: Slot index
            slot_name: Slot name
            mesh_name: Name of the mesh (for confidence scoring)
    
        Returns:
            SlotAnalysis dict
        """
        # Get material path (cleaned)
        source_material_path = None
        if slot_material:
            source_material_path = slot_material.get_path_name()
            if '.' in source_material_path:
                source_material_path = source_material_path.split('.')[0]
    
        slot_analysis = {
            'slot_index': slot_index,
            'slot_name': slot_name,
            'source_material_path': source_material_path,
            'packing_mode': 'STANDARD',
            'is_compatible': False,
            'skip_reason': None,
            'texture_matches': {},
            'has_conflicts': False,
            'conflict_types': [],
            'skipped_channels': {},
            'total_issues': 0
        }

        if not slot_material:
            slot_analysis['skip_reason'] = 'No material assigned'
            slot_analysis['total_issues'] = 1  # Incompatibility issue
            return slot_analysis
    
        # Get textures from material
        raw_textures = BPMaterialOptimizer.get_material_texture_dependencies(slot_material)
        if not raw_textures:
            slot_analysis['skip_reason'] = 'No textures found in material'
            slot_analysis['total_issues'] = 1  # Incompatibility issue
            return slot_analysis

        # Detect packing mode
        packing_mode = BPMaterialOptimizer.detect_texture_packing_mode(raw_textures)
        slot_analysis['packing_mode'] = packing_mode

        if packing_mode == 'RMA':
            slot_analysis['skip_reason'] = 'RMA packed textures (incompatible channel order)'
            slot_analysis['total_issues'] = 1  # Incompatibility issue
            return slot_analysis

        if packing_mode == 'RAM':
            slot_analysis['skip_reason'] = 'RAM packed textures (incompatible channel order)'
            slot_analysis['total_issues'] = 1  # Incompatibility issue
            return slot_analysis

        # Select master material based on packing mode
        if packing_mode == 'ORM':
            if not master_material_orm:
                slot_analysis['skip_reason'] = 'ORM textures detected but no ORM material provided'
                slot_analysis['total_issues'] = 1  # Incompatibility issue
                return slot_analysis
            selected_master = master_material_orm
        else:
            selected_master = master_material

        # Check compatibility
        is_compatible, reason = BPMaterialOptimizer.check_material_compatibility(slot_material, selected_master)
        slot_analysis['is_compatible'] = is_compatible
        if not is_compatible:
            slot_analysis['skip_reason'] = f'Incompatible: {reason}'
            slot_analysis['total_issues'] = 1  # Incompatibility issue
            return slot_analysis
    
        # Get all texture candidates with context for confidence scoring
        include_orm = (packing_mode == 'ORM')
        material_name = slot_material.get_name() if slot_material else None
        context_names = [mesh_name, material_name, slot_name]
        candidates = BPMaterialOptimizer.get_all_texture_candidates(raw_textures, include_orm=include_orm, context_names=context_names)
    
        slot_analysis['texture_matches'] = candidates
    
        # Build skipped_channels: false for each detected channel (user can set to true to skip)
        slot_analysis['skipped_channels'] = {tex_type: False for tex_type in candidates.keys()}
    
        # Check for conflicts (more than one candidate per type)
        for tex_type, tex_list in candidates.items():
            if len(tex_list) > 1:
                slot_analysis['has_conflicts'] = True
                slot_analysis['conflict_types'].append(tex_type)

        # Calculate total_issues: number of conflicts (incompatibilities already handled above)
        slot_analysis['total_issues'] = len(slot_analysis['conflict_types'])

        return slot_analysis

    @staticmethod
    def _analyze_mesh(
        mesh: unreal.StaticMesh,
        master_material,
        master_material_orm
    ) -> Dict[str, Any]:
        """Analyze a single mesh for texture conflicts.
    
        Args:
            mesh: The static mesh to analyze
            master_material: Standard master material
            master_material_orm: ORM master material (optional)
    
        Returns:
            MeshAnalysis dict
        """
        mesh_path = mesh.get_path_name()
        if '.' in mesh_path:
            mesh_path = mesh_path.split('.')[0]
    
        mesh_name = mesh.get_name()
        mesh_analysis = {
            'mesh_name': mesh_name,
            'mesh_path': mesh_path,
            'slots': [],
            'has_any_conflicts': False,
            'skip': False,
            'has_any_incompatibility': False,
            'total_issues': 0
        }

        material_slots = BPMaterialOptimizer.get_mesh_material_slots(mesh)
        if not material_slots:
            return mesh_analysis

        for slot in material_slots:
            slot_analysis = BPMaterialOptimizer._analyze_slot(
                slot['material'],
                master_material,
                master_material_orm,
                slot['index'],
                slot['slot_name'],
                mesh_name
            )
            mesh_analysis['slots'].append(slot_analysis)

            if slot_analysis['has_conflicts']:
                mesh_analysis['has_any_conflicts'] = True

            if not slot_analysis['is_compatible']:
                mesh_analysis['has_any_incompatibility'] = True
                # don't add the number of issues if the slot is incompatible, it won't be processed 
                return mesh_analysis

            # Sum up slot issues
            mesh_analysis['total_issues'] += slot_analysis['total_issues']
    
        return mesh_analysis


    # =============================================================================
    # Helper Functions
    # =============================================================================

    @staticmethod
    def normalize_content_path(path: str) -> str:
        """Normalize a content path to ensure it starts with /Game."""
        if not path:
            return path
        if not path.startswith("/"):
            path = "/" + path
        if not path.startswith("/Game"):
            path = "/Game" + path
        return path

    @staticmethod
    def ensure_folder_exists(folder_path: str) -> bool:
        """Create a content folder if it doesn't exist.
    
        Args:
            folder_path: Content path like "/Game/Materials/Instances"
    
        Returns:
            True if folder exists or was successfully created
        """
        folder_path = BPMaterialOptimizer.normalize_content_path(folder_path)

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
    @staticmethod
    def get_selected_static_meshes() -> list:
        """Get all currently selected static mesh assets from Content Browser."""
        editor_util = unreal.EditorUtilityLibrary()
        selected_assets = editor_util.get_selected_assets()

        static_meshes = []
        for asset in selected_assets:
            if isinstance(asset, unreal.StaticMesh):
                static_meshes.append(asset)

        return static_meshes

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return BPMaterialOptimizer.levenshtein_distance(s2, s1)

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

    @staticmethod
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

    @staticmethod
    def get_mesh_base_name(mesh_name: str) -> str:
        """Extract the base name from a mesh, removing common prefixes."""
        base = mesh_name
        for prefix in ["SM_", "S_", "Mesh_", "M_"]:
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        return base

    @staticmethod
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
        base_name = BPMaterialOptimizer.get_mesh_base_name(mesh_name)
        base_name_lower = base_name.lower()
        matched_textures = {}

        # First pass: exact matching (base name contained in texture base name)
        for texture in textures:
            tex_name = texture.get_name()
            tex_base = BPMaterialOptimizer.get_texture_base_name(tex_name, patterns)
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
            tex_base = BPMaterialOptimizer.get_texture_base_name(tex_name, patterns)
            tex_base_lower = tex_base.lower()

            # Calculate similarity
            distance = BPMaterialOptimizer.levenshtein_distance(base_name_lower, tex_base_lower)
            max_len = max(len(base_name_lower), len(tex_base_lower))
            if max_len == 0:
                continue

            similarity = 1.0 - (distance / max_len)
            tex_type = BPMaterialOptimizer.match_texture_to_type(tex_name, patterns)

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

    @staticmethod
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

    @staticmethod
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
            package_path = BPMaterialOptimizer.normalize_content_path(output_folder)
            # Ensure the output folder exists, create if needed
            if not BPMaterialOptimizer.ensure_folder_exists(package_path):
                unreal.log_error(f"  Cannot create output folder: {package_path}")
                return None, False
        else:
            # Use same folder as mesh
            package_path = "/".join(mesh_path.split("/")[:-1])

        # Derive MI name from source material
        mi_name = BPMaterialOptimizer.derive_mi_name_from_material(source_material)
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

    @staticmethod
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

    @staticmethod
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
            param_name = BPMaterialOptimizer.find_texture_parameter_name(material_instance, tex_type, param_names)

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

    @staticmethod
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


    @unreal.ufunction(static=True, params=[str, str], ret=str, meta=dict(Category="Material Optimizer"))
    def analyze_selected_meshes(
        master_material_path: str,
        master_material_orm_path: str
    ) -> str:
        """Analyze selected static meshes and return a JSON report of their materials and textures.

        This is Phase 1 of the two-phase workflow. It performs read-only analysis without
        modifying any assets. The returned JSON can be passed to optimize_materials_from_analysis()
        after optional user modifications to resolve conflicts or skip certain items.

        The analysis examines each mesh's material slots, finds texture dependencies,
        categorizes textures by type (BaseColor, Normal, Roughness, etc.), detects
        ORM/ARM/RMA packing modes, and calculates confidence scores for texture matches.

        Args:
            master_material_path: Content path to the standard master material
                (e.g., "/Game/Materials/M_Master"). Used to check material compatibility.
            master_material_orm_path: Content path to the ORM variant master material
                (e.g., "/Game/Materials/M_Master_ORM"). Pass empty string if not using ORM.
                Required only if meshes use ORM/ARM packed textures.

        Returns:
            JSON string with the following structure:
            {
                "error": str or null,           // Error message if analysis failed
                "meshes": [                     // Array of mesh analysis results
                    {
                        "mesh_name": str,
                        "mesh_path": str,
                        "slots": [              // Material slot analyses
                            {
                                "slot_index": int,
                                "slot_name": str,
                                "source_material_path": str,
                                "packing_mode": "STANDARD" | "ORM",
                                "is_compatible": bool,
                                "skip_reason": str or null,
                                "texture_matches": {
                                    "BaseColor": [{"name", "path", "confidence", "selected"}, ...],
                                    "Normal": [...],
                                    ...
                                },
                                "has_conflicts": bool,
                                "conflict_types": [str],
                                "skipped_channels": {"BaseColor": false, ...},
                                "total_issues": int
                            }
                        ],
                        "has_any_conflicts": bool,
                        "skip": bool,
                        "has_any_incompatibility": bool,
                        "total_issues": int
                    }
                ],
                "total_issues": int,            // Sum of all mesh total_issues (conflicts + incompatibilities)
                "meshes_with_conflicts": int,
                "can_auto_process": bool        // True if no conflicts require user resolution
            }
        """
        # Load master materials
        master_material = unreal.EditorAssetLibrary.load_asset(master_material_path)
        if not master_material:
            return json.dumps({
                'error': f'Master material not found: {master_material_path}',
                'meshes': [],
                'total_issues': 0,
                'meshes_with_conflicts': 0,
                'can_auto_process': False
            })

        master_material_orm = None
        if master_material_orm_path:
            master_material_orm = unreal.EditorAssetLibrary.load_asset(master_material_orm_path)
            if not master_material_orm:
                return json.dumps({
                    'error': f'ORM master material not found: {master_material_orm_path}',
                    'meshes': [],
                    'total_issues': 0,
                    'meshes_with_conflicts': 0,
                    'can_auto_process': False
                })

        # Get selected meshes
        meshes = BPMaterialOptimizer.get_selected_static_meshes()
        if not meshes:
            return json.dumps({
                'error': 'No static meshes selected',
                'meshes': [],
                'total_issues': 0,
                'meshes_with_conflicts': 0,
                'can_auto_process': False
            })

        # Analyze each mesh
        result = {
            'meshes': [],
            'total_issues': 0,
            'meshes_with_conflicts': 0,
            'can_auto_process': True
        }

        for mesh in meshes:
            mesh_analysis = BPMaterialOptimizer._analyze_mesh(mesh, master_material, master_material_orm)
            result['meshes'].append(mesh_analysis)

            # Sum up mesh issues (which includes all slot issues)
            result['total_issues'] += mesh_analysis['total_issues']

            if mesh_analysis['has_any_conflicts']:
                result['meshes_with_conflicts'] += 1
                result['can_auto_process'] = False

        unreal.log(f"Analysis complete: {len(meshes)} meshes, {result['total_issues']} issues")
        return json.dumps(result)
    
    @unreal.ufunction(static=True, params=[str, str, str, str, bool], ret=str, meta=dict(Category="Material Optimizer"))
    def optimize_materials_from_analysis(
        analysis_json: str,
        master_material_path: str,
        master_material_orm_path: str,
        output_folder: str,
        overwrite: bool
    ) -> str:
        """Process meshes and create Material Instances based on analysis results.

        This is Phase 2 of the two-phase workflow. It takes the JSON output from
        analyze_selected_meshes() and creates optimized Material Instances for each
        mesh's material slots, assigning the appropriate textures.

        The analysis JSON can be modified before calling this function to:
        - Resolve texture conflicts by setting 'selected': true on preferred candidates
        - Skip entire meshes by setting 'skip': true on the mesh entry
        - Skip specific texture channels by setting them to true in 'skipped_channels'

        For each compatible material slot, this function:
        1. Creates a new MaterialInstanceConstant parented to the master material
        2. Assigns textures to the appropriate material parameters
        3. Assigns the new MI to the mesh's material slot

        Skipping Logic:
        - Meshes with 'skip': true are skipped entirely
        - Slots with 'is_compatible': false are skipped
        - Slots with unresolved conflicts (multiple candidates, none selected) are skipped
        - Channels in 'skipped_channels' set to true are not assigned

        Args:
            analysis_json: JSON string from analyze_selected_meshes(), potentially modified
                by the user to resolve conflicts. Each texture candidate has 'selected': bool
                and textures with confidence >= 1.0 are auto-selected during analysis.
            master_material_path: Content path to the standard master material
                (e.g., "/Game/Materials/M_Master"). New MIs will be parented to this.
            master_material_orm_path: Content path to the ORM variant master material.
                Pass empty string if not using ORM. Used for slots with ORM packed textures.
            output_folder: Content path where Material Instances will be created
                (e.g., "/Game/Materials/Instances"). Pass empty string to save next to mesh.
            overwrite: If true, existing Material Instances with the same name will be
                deleted and recreated. If false, existing MIs are reused without modification.

        Returns:
            JSON string with processing results:
            {
                "error": str or null,       // Error message if processing failed
                "processed": int,           // Number of meshes successfully processed
                "skipped": int,             // Number of meshes skipped
                "saved_count": int          // Number of new MIs created and saved
            }

        Note:
            Static meshes are NOT auto-saved after material assignment. Save them manually
            from the Content Browser (Ctrl+Shift+S) to persist the material slot changes.
        """
        unreal.log("")
        unreal.log("=" * 50)
        unreal.log("MATERIAL ASSIGNER (With User Selections)")
        unreal.log("=" * 50)
    
        # Parse analysis result
        try:
            analysis = json.loads(analysis_json)
        except json.JSONDecodeError as e:
            error_msg = f'Invalid JSON: {str(e)}'
            unreal.log_error(error_msg)
            return json.dumps({'error': error_msg, 'processed': 0, 'skipped': 0})
    
        # Load master materials
        master_material = unreal.EditorAssetLibrary.load_asset(master_material_path)
        if not master_material:
            error_msg = f'Master material not found: {master_material_path}'
            unreal.log_error(error_msg)
            return json.dumps({'error': error_msg, 'processed': 0, 'skipped': 0})
    
        master_material_orm = None
        if master_material_orm_path:
            master_material_orm = unreal.EditorAssetLibrary.load_asset(master_material_orm_path)
            if not master_material_orm:
                error_msg = f'ORM master material not found: {master_material_orm_path}'
                unreal.log_error(error_msg)
                return json.dumps({'error': error_msg, 'processed': 0, 'skipped': 0})
    
        assets_to_save = []
        processed = 0
        skipped = 0
    
        meshes = analysis.get('meshes', [])
        num_meshes = len(meshes)
    
        with unreal.ScopedSlowTask(num_meshes, "Processing Materials...") as slow_task:
            slow_task.make_dialog(True)
    
            for mesh_data in meshes:
                if slow_task.should_cancel():
                    unreal.log_warning("Operation cancelled by user")
                    break
    
                mesh_path = mesh_data.get('mesh_path')
                mesh_name = mesh_data.get('mesh_name', 'Unknown')
                has_any_conflicts = mesh_data.get('has_any_conflicts', False)
                skip_mesh = mesh_data.get('skip', False)
    
                # Check if user wants to skip this mesh
                if skip_mesh:
                    unreal.log(f"  SKIPPED: User marked mesh to skip")
                    skipped += 1
                    continue
    
                slow_task.enter_progress_frame(1, f"Processing: {mesh_name}")
                unreal.log(f"\n[{mesh_name}]")
    
                # Load the mesh
                mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
                if not mesh or not isinstance(mesh, unreal.StaticMesh):
                    unreal.log_warning(f"  SKIPPED: Mesh not found or invalid")
                    skipped += 1
                    continue
    
                material_slots = BPMaterialOptimizer.get_mesh_material_slots(mesh)
                slots_processed = 0
    
                for slot_data in mesh_data.get('slots', []):
                    slot_index = slot_data.get('slot_index', 0)
                    slot_name = slot_data.get('slot_name', '')
                    is_compatible = slot_data.get('is_compatible', False)
                    skip_reason = slot_data.get('skip_reason')
                    has_conflicts = slot_data.get('has_conflicts', False)
                    packing_mode = slot_data.get('packing_mode', 'STANDARD')
                    texture_matches = slot_data.get('texture_matches', {})
    
                    slot_label = f"[Slot {slot_index}: {slot_name}]"
    
                    # Skip if not compatible
                    if not is_compatible:
                        reason = skip_reason or 'Not compatible'
                        unreal.log_warning(f"  {slot_label} SKIPPED: {reason}")
                        continue
    
                    # Get selected textures from texture_matches
                    matched = {}
                    unresolved_conflicts = []
                    skipped_channels = slot_data.get('skipped_channels', {})
    
                    for tex_type, tex_list in texture_matches.items():
                        # Skip channels marked by user
                        if skipped_channels.get(tex_type, False):
                            unreal.log(f"    {tex_type}: SKIPPED (user)")
                            continue
    
                        selected_tex = None
                        for tex_entry in tex_list:
                            if tex_entry.get('selected', False):
                                selected_tex = tex_entry
                                break
    
                        if selected_tex:
                            # Load the texture
                            texture = unreal.EditorAssetLibrary.load_asset(selected_tex['path'])
                            if texture and isinstance(texture, unreal.Texture):
                                matched[tex_type] = texture
                            else:
                                unreal.log_warning(f"  Texture not found: {selected_tex['path']}")
                        elif len(tex_list) > 1:
                            # Conflict not resolved
                            unresolved_conflicts.append(tex_type)
    
                    # Skip if there are unresolved conflicts
                    if unresolved_conflicts:
                        unreal.log_warning(f"  {slot_label} SKIPPED: Unresolved conflicts for: {', '.join(unresolved_conflicts)}")
                        continue
    
                    if not matched:
                        unreal.log_warning(f"  {slot_label} SKIPPED: No textures selected")
                        continue
    
                    # Find the actual slot material
                    slot = None
                    for s in material_slots:
                        if s['index'] == slot_index:
                            slot = s
                            break
    
                    if not slot or not slot['material']:
                        unreal.log_warning(f"  {slot_label} SKIPPED: No material in slot")
                        continue
    
                    slot_material = slot['material']
    
                    # Select master material based on packing mode
                    if packing_mode == 'ORM':
                        if not master_material_orm:
                            unreal.log_warning(f"  {slot_label} SKIPPED: ORM textures but no ORM material")
                            continue
                        selected_master = master_material_orm
                        param_names = MATERIAL_PARAMETER_NAMES_ORM
                    else:
                        selected_master = master_material
                        param_names = MATERIAL_PARAMETER_NAMES
    
                    # Create material instance
                    mi, needs_assignment = BPMaterialOptimizer.create_material_instance(
                        mesh, selected_master, slot_material, output_folder, overwrite
                    )
                    if not mi:
                        continue
    
                    unreal.log(f"  {slot_label} Packing: {packing_mode}")
    
                    # Assign textures
                    if needs_assignment:
                        unreal.log("  Assigning textures:")
                        BPMaterialOptimizer.assign_textures_to_material(mi, matched, param_names)
                        assets_to_save.append(mi)
    
                    # Assign to mesh
                    BPMaterialOptimizer.assign_material_to_mesh(mesh, mi, slot_index)
    
                    slots_processed += 1
    
                if slots_processed > 0:
                    processed += 1
                else:
                    skipped += 1
    
        # Save new material instances
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
    
        return json.dumps({
            'processed': processed,
            'skipped': skipped,
            'saved_count': len(assets_to_save)
        })
 
