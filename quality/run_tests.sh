#!/bin/bash

# Get root directory :
ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${ROOTDIR}

# NOTE : tests/ ne contient volontairement aucun test pytest à ce
# stade (scripts de démonstration -> examples/, exploration -> tests/exploratory/,
# voir REFERENCE_TECHNIQUE.md §5.4 pour la justification). pytest ne collectera
# donc rien ici tant qu'une vraie suite assertive n'aura pas été ajoutée sous
# tests/ — laissé en l'état (pas cassé, juste vide) pour ne pas avoir à
# retoucher ce script quand cette suite sera écrite.
coverage run -m pytest ${ROOTDIR}/../tests
