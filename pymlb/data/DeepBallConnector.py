from abc import ABC
from typing import Dict, Optional
import urllib.request as reqlib
import urllib.parse as parse
import json
import re
from os.path import isfile, isdir, join
import gzip
import warnings
from hashlib import sha1


class DeepBallConnector(ABC):
    @staticmethod
    def _slugify(value):
        """
        Normalizes string, converts to lowercase, removes non-alpha characters,
        and converts spaces to hyphens.
        """
        value = re.sub('>', 'gt', value.strip().lower())
        value = re.sub('<', 'lt', value)
        value = re.sub('=', 'eq', value)
        value = re.sub('!', 'nt', value)
        value = re.sub('[^\w\s-]', '', value).strip().lower()
        return re.sub('[-\s]+', '-', value)

    def __init__(self, connector: 'DeepBallConnector' = None, token: str = None, cache_directory: str = None):
        if connector is None and token is None:
            raise ValueError("No token was specified. Missing both 'connector' and 'token' parameters.")

        self.token = (token or connector.token)
        self.cache_directory = (cache_directory or connector.cache_directory)

    def _query(self, resource: str, parameters: Optional[Dict[str, object]] = None, cache: bool = False,
               verb: str = "GET") -> Dict[str, object]:
        if cache and verb != "GET":
            warnings.warn("Caching is not valid with non-GET requests. Turning cache flag off.")
            cache = False

        if cache and self.cache_directory is None:
            warnings.warn("Caching was requested but there was no cache directory given. Turning cache flag off.")
            cache = False
        elif cache and not isdir(self.cache_directory):
            warnings.warn("Caching was requested but the cache directory does not exist. Turning cache flag off.")
            cache = False

        # prepare the request
        headers = {
            'Authorization': 'Token token="' + self.token + '"',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'Content-Type': 'application/json',
        }
        query_string = (
            "" if verb != "GET" or parameters is None else "?" + parse.quote_plus(
                json.dumps(parameters, separators=(',', ':'), sort_keys=True)))
        url = resource + query_string

        # see if it's cached
        this_file_name = verb + self._slugify(url)
        if len(this_file_name) > 160:
            this_file_name = sha1(this_file_name.encode('utf-8')).hexdigest()
        cache_file = "" if self.cache_directory is None else join(self.cache_directory, this_file_name + ".json")
        if cache and isfile(cache_file):
            # it is cached
            with open(cache_file, 'r') as file:
                string_contents = file.read()
        else:
            # either it's not cached or they don't want to cache. pull it fresh
            data = (None if verb == "GET" else bytearray(json.dumps(parameters, separators=(',', ':')), "utf-8"))
            req = reqlib.Request("https://[hidden]/" + url, data=data, headers=headers, method=verb)
            response = reqlib.urlopen(req)
            if response.info().get('Content-Encoding') == 'gzip':
                string_contents = gzip.decompress(response.read()).decode("utf-8")
            else:
                string_contents = response.read().decode("utf-8")

            # cache this if they want
            if cache:
                with open(cache_file, "w") as text_file:
                    text_file.write(string_contents)

        return json.loads(string_contents)

    def _get(self, resource: str, parameters: Optional[Dict[str, object]] = None, cache: bool = False) -> Dict[
        str, object]:
        return self._query(resource, parameters, cache, "GET")

    def _put(self, resource: str, parameters: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        return self._query(resource, parameters, False, "PUT")

    def _post(self, resource: str, parameters: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        return self._query(resource, parameters, False, "POST")

    def _patch(self, resource: str, parameters: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        return self._query(resource, parameters, False, "PATCH")

    def _delete(self, resource: str, parameters: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        return self._query(resource, parameters, False, "DELETE")
