# Convert outputs from extract_repos to miniCTX format

import argparse
import glob
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    'input_dir', type=str,
    help='Input directory containing output from extract_repos.py (e.g. Examples/Mathlib)'
)
parser.add_argument(
    'out_file', type=argparse.FileType('w'),
    help='Output in miniCTX format as .jsonl'
)
args = parser.parse_args()


repo_premises = set()
for filename in glob.glob(os.path.join(args.input_dir, 'Premises/*.jsonl')):
    with open(filename) as file:
        for line in file:
            data = json.loads(line)
            repo_premises.add(data['name'])

theorems = []
for filename in glob.glob(os.path.join(args.input_dir, 'FullProof/*.jsonl')):
    dependency_map = {}
    premises_filename = filename.replace('FullProof', 'Premises', 1)
    with open(premises_filename) as premises_file:
        for line in premises_file:
            data = json.loads(line)
            dependency_map[data['name']] = data['dependents']
    with open(filename) as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            context = data['srcUpToDecl']
            name = data['declName']
            proof = data['proof']

            if name in dependency_map:
                dependents = dependency_map[name]
                has_in_file_dependency = any(d['name'] in dependency_map for d in dependents)
                has_repo_dependency = any(d['name'] in repo_premises for d in dependents)
            else:
                # not available (because of some probably fixable error)
                has_in_file_dependency = None
                has_repo_dependency = None

            theorems.append({
                'srcContext': context,
                'theoremStatement': data['decl'],
                'theoremName': name,
                # 'theoremId': data['declId'],

                'fileCreated': None,  # TODO
                'theoremCreated': None,  # TODO
                'file': data['file'],
                'module': data['module'],
                'jsonFile': os.path.basename(filename),

                'positionMetadata': {
                    'lineInFile': 1 + context.count('\n'),
                    'tokenPositionInFile': len(context),
                    # The JSONL is collected in order (see full_proof_training_data)
                    'theoremPositionInFile': i,
                },
                'dependencyMetadata': {
                    'inFilePremises': has_in_file_dependency,
                    'repositoryPremises': has_repo_dependency,
                },
                'proofMetadata': {
                    'hasProof': 'sorry' not in proof,
                    'proof': proof,
                    'proofType': 'tactic' if proof.startswith(':= by') else 'term',
                    'proofLengthLines': proof.count('\n'),
                    'proofLengthTokens': len(proof),
                }
            })

for theorem in theorems:
    args.out_file.write(json.dumps(theorem))
    args.out_file.write('\n')
args.out_file.close()
