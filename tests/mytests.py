from main import client

def test_old_man(client):
    response = client.get(
        "/get-age-by-photo",
        query_string={
            'url': 'https://indasil.club/uploads/posts/2022-11/1669494194_45-indasil-club-p-portret-starogo-cheloveka-instagram-51.jpg',
        }
    )
    assert response.status_code == 200
    assert response.json == {
        'Age': 'more than 70'}

def test_young_man(client):
    response = client.get(
        "/get-age-by-photo",
        query_string={
            'url': 'https://luckclub.ru/images/luckclub/2019/03/newborn-baby-asleep-1.jpg',
        }
    )
    assert response.status_code == 200
    assert response.json == {
        'Age': '0-2'}
def test_empty_query(client):
    response = client.get(
        "/get-age-by-photo",
        query_string={
            'url': '',
        }
    )
    assert response.status_code == 404

# python -m pytest app/tests/test.py

