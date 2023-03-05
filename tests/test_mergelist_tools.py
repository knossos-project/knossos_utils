import pytest

from knossos_utils.mergelist_tools import objects_from_mergelist, subobject_map_from_mergelist

inps = ['''\
1 0 0 1234567890123456789
0 0 0


'''
,'''\
3150 1 1 5352 3297 3971 4104 2654 4153 4046 2257 3624 3150 4100
5333 5267 2880
mito
whatever one may want to add here
3151 1 0 2257 2654 2663 3018 3023 3029 3150 3294 3297 3298 3363
1024 2048 4096 16 25 222
neuron
proofread merge
''']
sv_set_list = [
    [{1234567890123456789}]
    ,[{5352,3297,3971,4104,2654,4153,4046,2257,3624,3150,4100},{2257,2654,2663,3018,3023,3029,3150,3294,3297,3298,3363}]
]
sv_obj_map = [
    {1234567890123456789:1}
    ,{2257:3151,2654:3151,2663:3151,3018:3151,3023:3151,3029:3151,3150:3151,3294:3151,3297:3151,3298:3151,3363:3151,3624:3150,3971:3150,4046:3150,4100:3150,4104:3150,4153:3150,5352:3150}
]
funcs = [objects_from_mergelist, subobject_map_from_mergelist]
outs_list = [sv_set_list, sv_obj_map]

@pytest.mark.parametrize('func,test_input,expected_output', [ (f,i,o) for f,outs in zip(funcs, outs_list) for i,o in zip(inps, outs) ] )
def test(func, test_input, expected_output):
    assert func(test_input) == expected_output
