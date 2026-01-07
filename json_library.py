import unreal
import json

@unreal.uclass()
class BPJsonLibrary(unreal.BlueprintFunctionLibrary):

    @staticmethod
    def set_json_value(json_string, json_path, value):
        """
        Set a value at any path in a JSON structure.
        
        json_path: dot notation, use [0] for array indices. e.g. "users[2].settings.enabled"
        value_json: the value as a JSON string, e.g. '"hello"' or '42' or '{"nested": true}' or '[1,2,3]'
        """
        unreal.log(f"{json_path}: {value}")
        data = json.loads(json_string)

        BPJsonLibrary._set_at_path(data, json_path, value)
        return json.dumps(data)

    @unreal.ufunction(static=True, params=[str, str, str], ret=str, meta=dict(Category="JSON Utils"))
    def set_string_json_value(json_string, json_path, value_json):
        return BPJsonLibrary.set_json_value(json_string, json_path, value_json)

    @unreal.ufunction(static=True, params=[str, str, bool], ret=str, meta=dict(Category="JSON Utils"))
    def set_bool_json_value(json_string, json_path, value_json):
        return BPJsonLibrary.set_json_value(json_string, json_path, value_json)

    @unreal.ufunction(static=True, params=[str, str, int], ret=str, meta=dict(Category="JSON Utils"))
    def set_int_json_value(json_string, json_path, value_json):
        return BPJsonLibrary.set_json_value(json_string, json_path, value_json)

    @unreal.ufunction(static=True, params=[str, str, float], ret=str, meta=dict(Category="JSON Utils"))
    def set_float_json_value(json_string, json_path, value_json):
        return BPJsonLibrary.set_json_value(json_string, json_path, value_json)
    
    @unreal.ufunction(static=True, params=[str, str], ret=str, meta=dict(Category="JSON Utils"))
    def get_json_value(json_string, json_path):
        """
        Get a value at any path, returned as JSON string.
        """
        data = json.loads(json_string)
        result = BPJsonLibrary._get_at_path(data, json_path)
        return json.dumps(result)
    
    @unreal.ufunction(static=True, params=[str, str, str], ret=str, meta=dict(Category="JSON Utils"))
    def append_to_json_array(json_string, array_path, value_json):
        """
        Append a value to an array at the given path.
        """
        data = json.loads(json_string)
        value = json.loads(value_json)
    
        arr = BPJsonLibrary._get_at_path(data, array_path)
        arr.append(value)
        return json.dumps(data)
    
    @unreal.ufunction(static=True, params=[str, str, int], ret=str, meta=dict(Category="JSON Utils"))
    def remove_from_json_array(json_string, array_path, index):
        """
        Remove item at index from array at path.
        """
        data = json.loads(json_string)
        arr = BPJsonLibrary._get_at_path(data, array_path)
        arr.pop(index)
        return json.dumps(data)
    
    @staticmethod
    def _parse_path(path):
        """Parse 'foo.bar[0].baz' into [('foo', None), ('bar', 0), ('baz', None)]"""
        import re
        tokens = []
        for part in path.split('.'):
            match = re.match(r'(\w+)(?:\[(\d+)\])?', part)
            if match:
                key = match.group(1)
                idx = int(match.group(2)) if match.group(2) else None
                tokens.append((key, idx))
        return tokens
    
    @staticmethod
    def _get_at_path(data, path):
        tokens = BPJsonLibrary._parse_path(path)
        current = data
        for key, idx in tokens:
            current = current[key]
            if idx is not None:
                current = current[idx]
        return current
    
    @staticmethod
    def _set_at_path(data, path, value):
        tokens = BPJsonLibrary._parse_path(path)
        current = data
        for key, idx in tokens[:-1]:
            current = current[key]
            if idx is not None:
                current = current[idx]
    
        last_key, last_idx = tokens[-1]
        if last_idx is not None:
            current[last_key][last_idx] = value
        else:
            current[last_key] = value

unreal.log("JSON Library loaded")