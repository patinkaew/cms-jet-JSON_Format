import argparse
import json
from JetIDHelpers import create_jetId_correctionSet, write_correctionSet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create correctionlib-compatible JSON from JetID spec JSON file")
    parser.add_argument("jetID_spec_json_file", nargs=1, type=str, help="JetID spec JSON file")
    parser.add_argument("--description", "-d", nargs=1, type=str, required=False, default="Jet Identification Criteria", help="Overall description of JetID correctionSet", dest="description")
    parser.add_argument("--out_filename", "-o", nargs=1, type=str, required=False, default="jetID.json", help="Filename of output JSON", dest="out_filename")
    parser.add_argument("--write_gzip", action="store_true", default=True, help="whether to write .json.gz file in addition to .json file", dest="write_gzip")
    args = parser.parse_args()

    jetId_tasks = list()
    with open(args.jetID_spec_json_file[0]) as f:
        jetId_tasks = json.load(f)
    
    corrSet = create_jetId_correctionSet(jetId_tasks, description=args.description[0])
    write_correctionSet(corrSet, args.out_filename[0], args.write_gzip)