Installation:
- plugin dependency : Json Blueprint Utilities
- unpack the content of the repo in Content/AssetTools (or whatever name you want).
- Add this folder in you Config/DefaultEngine.ini like this
```
[/Script/PythonScriptPlugin.PythonScriptPluginSettings]
+AdditionalPaths=(Path="Content/AssetTools")
```

Usage
- right click on a static mesh (or on a selection of static meshes)
- choose "Scripted Asset Actions/Optimie Assets" in the context menu
- pick a 