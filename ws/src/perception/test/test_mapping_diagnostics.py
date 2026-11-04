"""Synthetic occupancy, geometry, correction and snapshot diagnostics regressions."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml
from PIL import Image

from perception.mapping_diagnostics import (compare_rasters, diagnose, load_trajectory,
                                             map_data, match_corrections, motion, occupancy)
from perception.save_slam_map import save_bundle


def make_map(tmp_path, yaw=0.):
    pixels = np.array([[0,205],[254,254]], np.uint8)
    Image.fromarray(pixels).save(tmp_path/'map.pgm')
    meta = {'image':'map.pgm','resolution':.05,'origin':[1.,2.,yaw],
            'mode':'trinary','negate':0,'free_thresh':.196,'occupied_thresh':.65}
    (tmp_path/'map.yaml').write_text(yaml.safe_dump(meta))
    return tmp_path/'map.yaml'


def test_occupancy_origin_rotation_flip_and_outside(tmp_path):
    path = make_map(tmp_path, np.pi/2)
    meta, _, free, occupied = map_data(path)
    local = np.array([[.025,.025],[.025,.075],[.075,.075],[.15,.15]])
    xy = local @ np.array([[0,1],[-1,0]]) + [1,2]
    tr = np.column_stack([np.arange(1,5),xy,np.zeros(4)])
    result = occupancy(tr,meta,free,occupied)
    assert [result[k]['count'] for k in ('free','occupied','unknown','outside')] == [1,1,1,1]
    assert all(v['fraction']==.25 for v in result.values())


def test_unknown_does_not_become_free(tmp_path):
    path=make_map(tmp_path)
    meta,_,free,occupied=map_data(path)
    assert not free[0,1] and not occupied[0,1]
    assert meta['resolution']==.05


def test_large_jumps_and_timestamp_matched_corrections():
    online=np.array([[1,0,0,0],[1.1,3,0,0],[2,3.1,0,0]],float)
    final=np.array([[1,0,0,0],[1.1,.1,0,0],[2,.2,0,0]],float)
    assert len(motion(online)['discontinuities'])==1
    assert not motion(final)['discontinuities']
    result=match_corrections(online,final)
    assert result['timestamp_matches']==3
    assert result['max_position_correction_m']==pytest.approx(2.9)
    final[:,0]+=10
    assert match_corrections(online,final)['max_position_correction_m'] is None


def test_raster_encoding_or_geometry_difference(tmp_path):
    a=make_map(tmp_path)
    child=tmp_path/'other';child.mkdir();b=make_map(child)
    assert compare_rasters(a,b)['different_pixels']==0
    pixels=np.asarray(Image.open(child/'map.pgm')).copy();pixels[0,0]=254
    Image.fromarray(pixels).save(child/'map.pgm')
    assert compare_rasters(a,b)['different_pixels']==1
    meta=yaml.safe_load(b.read_text());meta['origin'][0]+=1;b.write_text(yaml.safe_dump(meta))
    assert compare_rasters(a,b)['different_pixels'] is None


def test_reports_failures_and_missing_logs_without_inventing_data(tmp_path):
    path=make_map(tmp_path)
    trajectory=np.array([[1,1.025,2.025,0],[2,1.025,2.075,0]])
    np.savetxt(tmp_path/'trajectory.csv',trajectory,delimiter=',')
    log=tmp_path/'mapper.log'
    log.write_text('Message Filter dropping: queue is full\nWARN Bounding to maximum number available\n')
    report=diagnose(path,tmp_path/'trajectory.csv',mapping_log=log)
    assert report['mapping_log']['queue_full_drops']==1
    assert report['mapping_log']['thread_limit_warnings']==1
    assert report['follower_log']['warnings'] is None
    assert report['online']['occupancy']['occupied']['count']==1
    json.dumps(report,allow_nan=False)


def test_invalid_trajectory_and_map_fail(tmp_path):
    np.savetxt(tmp_path/'bad.csv',[[1,0,0,0],[1,1,0,0]],delimiter=',')
    with pytest.raises(ValueError):load_trajectory(tmp_path/'bad.csv')
    path=make_map(tmp_path);meta=yaml.safe_load(path.read_text());meta['resolution']=0
    path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError):map_data(path)


def test_stale_history_distinguished_from_poor_final_geometry(tmp_path, monkeypatch):
    import planning.centerline
    monkeypatch.setattr(planning.centerline, 'extract_centerline', lambda *args: None)
    path=make_map(tmp_path)
    online=np.array([[1,1.025,2.075,0],[2,1.025,2.025,0]])
    final=np.array([[1,1.025,2.025,0],[2,1.075,2.025,0]])
    np.savetxt(tmp_path/'online.csv',online,delimiter=',')
    np.savetxt(tmp_path/'final.csv',final,delimiter=',')
    report=diagnose(path,tmp_path/'online.csv',tmp_path/'final.csv',mapping_log=tmp_path/'absent.log')
    assert report['online']['occupancy']['occupied']['count']==1
    assert report['final']['occupancy']['occupied']['count']==0
    assert report['planner_geometry']['passed'] is True
    assert report['mapping_log']['available'] is False
    assert any('alone does not establish' in message for message in report['interpretation'])
    def bad_geometry(*args): raise RuntimeError('loop does not follow the lap')
    monkeypatch.setattr(planning.centerline, 'extract_centerline', bad_geometry)
    report=diagnose(path,tmp_path/'online.csv',tmp_path/'final.csv')
    assert report['planner_geometry']['passed'] is False
    assert any('still rejects' in message for message in report['interpretation'])
