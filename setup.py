# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Setup.py for the nfg_transformer package."""

import setuptools

with open('requirements.txt', 'r') as reqs_f:
  requirements = reqs_f.read().split()

with open('LICENSE', 'r') as lf:
  LICENSE = lf.read()

setuptools.setup(
    name='nfg_transformer',
    version='0.1',
    description=(
        'Code for the ICLR 2024 publication "NfgTransformer: Equivariant'
        ' Representation Learning for Normal-form Games".'
    ),
    url='https://openreview.net/forum?id=4YESQqIys7',
    author='Google Deepmind',
    author_email='liusiqi@google.com',
    license=LICENSE,
    install_requires=requirements,
    packages=['nfg_transformer'],
    zip_safe=False,
)
